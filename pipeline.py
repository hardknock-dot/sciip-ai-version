"""
SCIIP - Pipeline Orchestrator
Run every time you get a fresh Sales Register export (optionally with an
Item Master export too):

    python pipeline.py path/to/sales_register.xlsx [path/to/item_master.xlsx] [commitments.xlsx] [part_transfer.xlsx]

Storage note: SQLite for now - swap to DuckDB later if needed, the pipeline
structure doesn't change either way.
"""
import sys
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd

from ingest import load_sales_register
from item_master import load_item_master, check_demand_coverage
from transaction_engine import reconstruct_full_register, reconstruct_consignment_lines
from intelligence import (build_part_monthly_demand, classify_demand_patterns, build_customer_profile,
                            build_part_yearly_trend, build_customer_yearly_trend,
                            build_part_customer_breakdown, build_part_customer_concentration,
                            build_part_intelligence_summary, add_relationship_health,
                            build_project_intelligence, build_risk_alerts,
                            build_customer_forecast_decomposition)
from forecasting import (generate_forecasts, backtest_accuracy, build_part_monthly_pivot, tune_forecast_parameters,
                           generate_forecast_horizon, build_part_forecast_pivot, build_part_occurrence_pivot,
                           reconcile_forecast_log, summarize_forecast_reconciliation)
from replenishment import build_replenishment_recommendations
from customer_commitments import load_customer_commitments, apply_customer_commitments
from part_transfer_integration import run_part_transfer_step, load_part_master

DB_PATH = Path(__file__).parent / "data" / "sciip.db"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Only these projects are in scope for SCIIP right now. Adjust any time -
# everything else in the pipeline stays the same.
PROJECT_ALLOWLIST = ["ETCH", "INOVA", "SP-203"]


def run_pipeline(input_filepath: str, item_master_filepath: str = None, customer_commitments_filepath: str = None,
                  db_path=None, output_dir=None, item_master_df: pd.DataFrame = None,
                  customer_commitments_df: pd.DataFrame = None, part_transfer_filepath: str = None):
    """
    db_path / output_dir default to the standard locations. The dashboard's
    refresh feature passes a staging db_path so a failed run can never leave
    the live database half-updated. item_master_df / customer_commitments_df
    let a caller reuse data already held in the database (e.g. when only a
    new Sales Register was uploaded) instead of re-reading a file.
    part_transfer_filepath is the optional part transfer list (Etch unique list); when omitted,
    the transfer list already stored in the database is reused to rebuild part_master.
    """
    db_path = Path(db_path) if db_path else DB_PATH
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    db_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{run_timestamp}] SCIIP pipeline starting")
    print(f"Input file: {input_filepath}")

    # 1. Ingest
    print("\n--- Step 1: Ingestion ---")
    clean_df, report = load_sales_register(input_filepath)
    print(report.summary())

    pre_filter_count = len(clean_df)
    clean_df = clean_df[clean_df["project"].isin(PROJECT_ALLOWLIST)].reset_index(drop=True)
    excluded_by_project = pre_filter_count - len(clean_df)
    print(f"\nProject filter: keeping {PROJECT_ALLOWLIST}")
    print(f"Rows excluded by project filter: {excluded_by_project} ({len(clean_df)} rows remain for the engine)")

    # 2. Reconstruct - consignment orders use a direct SO-vs-Invoice path,
    #    everything else uses the DO-matching Transaction State Engine
    print("\n--- Step 2: Transaction State Engine ---")
    is_consignment = clean_df["buy_ord_ref"].fillna("").str.strip().str.lower().str.startswith("skyline consignment")
    consignment_df = clean_df[is_consignment]
    normal_df = clean_df[~is_consignment]

    normal_lines = reconstruct_full_register(normal_df) if len(normal_df) else pd.DataFrame()
    consignment_lines = reconstruct_consignment_lines(consignment_df) if len(consignment_df) else pd.DataFrame()
    demand_lines = pd.concat([normal_lines, consignment_lines], ignore_index=True)

    print(f"Reconstructed {len(demand_lines)} original demand lines "
          f"({len(normal_lines)} standard + {len(consignment_lines)} consignment) "
          f"from {len(clean_df)} register rows "
          f"({demand_lines['status'].eq('CLOSED').sum()} closed, {demand_lines['status'].eq('OPEN').sum()} open)")

    # 3. Demand & Customer Intelligence
    print("\n--- Step 3: Demand & Customer Intelligence ---")
    monthly_demand = build_part_monthly_demand(demand_lines)
    pattern_classification = classify_demand_patterns(monthly_demand)
    customer_profile = build_customer_profile(demand_lines)
    part_yearly_trend = build_part_yearly_trend(demand_lines)
    customer_yearly_trend = build_customer_yearly_trend(demand_lines)
    pattern_counts = pattern_classification["demand_pattern"].value_counts().to_dict()
    print(f"Classified {len(pattern_classification)} project+part combinations: {pattern_counts}")
    print(f"Built behavior profiles for {len(customer_profile)} customers")
    print(f"Built year-wise trend for {len(part_yearly_trend)} project+part combinations "
          f"and {len(customer_yearly_trend)} customer+project combinations")

    tuned_params = tune_forecast_parameters(monthly_demand, pattern_classification, min_history=6)
    if len(tuned_params) > 0:
        pct_met_baseline = 100 * tuned_params["met_baseline"].mean()
        print(f"Tuned parameters for {len(tuned_params)} project+part combinations "
              f"({pct_met_baseline:.0f}% meet/beat their own naive baseline)")
    else:
        print("Not enough history yet to tune any parts - using defaults")

    forecasts = generate_forecasts(monthly_demand, pattern_classification, tuned_params=tuned_params, horizon_months=6)
    print(f"Generated forecasts for {len(forecasts)} project+part combinations")

    forecast_horizon = generate_forecast_horizon(monthly_demand, pattern_classification, tuned_params=tuned_params, num_months=6)
    forecast_qty_pivot = build_part_forecast_pivot(forecast_horizon)
    forecast_occurrence_pivot = build_part_occurrence_pivot(forecast_horizon)
    print(f"Built 6-month forecast horizon: {len(forecast_horizon)} part-month rows "
          f"({len(forecast_qty_pivot)} parts x 6 months)")

    customer_breakdown = build_part_customer_breakdown(demand_lines)
    customer_breakdown = add_relationship_health(customer_breakdown)
    concentration = build_part_customer_concentration(customer_breakdown)
    part_intelligence = build_part_intelligence_summary(pattern_classification, forecasts, concentration)
    print(f"Built customer-concentration context for {len(concentration)} project+part combinations")
    print(f"Consolidated Part Intelligence summary: {len(part_intelligence)} rows (dashboard-ready)")
    status_counts = customer_breakdown["relationship_status"].value_counts().to_dict()
    print(f"Relationship health: {status_counts}")

    forecast_decomposition = build_customer_forecast_decomposition(part_intelligence, customer_breakdown)
    print(f"Built customer forecast decomposition: {len(forecast_decomposition)} customer-part rows")

    if customer_commitments_filepath:
        customer_commitments = load_customer_commitments(customer_commitments_filepath)
        print(f"Loaded {len(customer_commitments)} customer-provided forecast commitments")
    elif customer_commitments_df is not None and len(customer_commitments_df) > 0:
        customer_commitments = customer_commitments_df.copy()
        print(f"Reusing {len(customer_commitments)} customer commitments already in the database")
    else:
        customer_commitments = pd.DataFrame(columns=["project", "part_code", "cust_name", "month", "committed_qty"])

    forecast_horizon_adjusted = apply_customer_commitments(forecast_horizon, forecast_decomposition, customer_commitments)
    n_adjusted = forecast_horizon_adjusted["adjustment_note"].notna().sum()
    if n_adjusted > 0:
        print(f"Applied customer commitments: {n_adjusted} part-month forecasts adjusted with real customer input")

    accuracy = backtest_accuracy(monthly_demand, pattern_classification, min_history=6)
    monthly_pivot = build_part_monthly_pivot(monthly_demand)
    if len(accuracy) > 0:
        avg_hit_rate = accuracy["occurrence_hit_rate_pct"].mean()
        print(f"Backtested accuracy for {len(accuracy)} project+part combinations (avg occurrence hit rate: {avg_hit_rate:.1f}%)")
    else:
        print("Not enough history yet to backtest any parts")

    project_intelligence = build_project_intelligence(demand_lines, pattern_classification, forecasts, customer_breakdown)
    print(f"Built Project Intelligence for {len(project_intelligence)} projects")

    risk_alerts = build_risk_alerts(demand_lines, concentration, part_yearly_trend, accuracy, tuned_params, monthly_demand)
    if len(risk_alerts) > 0:
        alert_counts = risk_alerts["alert_type"].value_counts().to_dict()
        print(f"Built Risk & Alerts: {len(risk_alerts)} total flags - {alert_counts}")
    else:
        print("Built Risk & Alerts: no flags triggered")

    # Forecast log: append THIS run's predictions to the immutable log
    print("\n--- Step 3b: Forecast Log & Reconciliation ---")
    today_str = datetime.now().strftime("%Y-%m-%d")
    new_log_rows = forecast_horizon.copy()
    new_log_rows["run_date"] = today_str
    try:
        conn_read = sqlite3.connect(db_path)
        existing_log = pd.read_sql("SELECT * FROM part_forecast_log", conn_read)
        conn_read.close()
    except Exception:
        existing_log = pd.DataFrame()
    forecast_log = pd.concat([existing_log, new_log_rows], ignore_index=True)
    forecast_log = forecast_log.drop_duplicates(subset=["project", "part_code", "forecast_month", "run_date"], keep="first")
    print(f"Forecast log: {len(existing_log)} prior entries + {len(new_log_rows)} new = {len(forecast_log)} total (deduplicated)")

    reconciliation = reconcile_forecast_log(forecast_log, monthly_demand)
    reconciliation_summary = summarize_forecast_reconciliation(reconciliation)
    if len(reconciliation) > 0:
        print(f"Reconciled {len(reconciliation)} logged predictions against real outcomes")
    else:
        print("No logged predictions old enough to reconcile yet")

    # Item Master + Replenishment (optional)
    print("\n--- Step 3c: Item Master & Replenishment ---")
    if item_master_filepath:
        item_master, im_report = load_item_master(item_master_filepath)
        print(im_report.summary())
    elif item_master_df is not None and len(item_master_df) > 0:
        item_master = item_master_df.copy()
        print(f"Reusing existing Item Master already in the database ({len(item_master)} parts)")
    else:
        item_master = None
        print("No Item Master file provided - replenishment will run with stock treated as unknown")

    replenishment = build_replenishment_recommendations(part_intelligence, demand_lines, risk_alerts, item_master=item_master)
    status_counts = replenishment["status"].value_counts().to_dict()
    print(f"Built Replenishment recommendations for {len(replenishment)} parts: {status_counts}")

    if item_master is not None:
        coverage = check_demand_coverage(item_master, demand_lines)
        print(f"Item Master coverage: {coverage['demand_parts_missing_count']} demand parts missing from Item Master; "
              f"{coverage['item_master_parts_with_no_demand_count']} of {coverage['item_master_total_parts']} "
              f"Item Master parts have no demand history")
        demand_missing_df = pd.DataFrame({"part_code": coverage["demand_parts_missing_from_item_master"]})
        im_no_demand_df = pd.DataFrame({"part_code": coverage["item_master_parts_with_no_demand"]})
    else:
        demand_missing_df = pd.DataFrame(columns=["part_code"])
        im_no_demand_df = pd.DataFrame(columns=["part_code"])

    # 4. Store (full replace each run, except the append-only forecast log)
    print("\n--- Step 4: Storage ---")
    conn = sqlite3.connect(db_path)

    def store(df, name, date_cols=None):
        d = df.copy()
        for col in d.columns:
            if pd.api.types.is_datetime64_any_dtype(d[col]):
                d[col] = d[col].dt.strftime("%Y-%m-%d")
        d["_computed_at"] = run_timestamp
        d.to_sql(name, conn, if_exists="replace", index=False)
        return len(d)

    clean_df_store = clean_df.copy()
    for col in clean_df_store.columns:
        if pd.api.types.is_datetime64_any_dtype(clean_df_store[col]):
            clean_df_store[col] = clean_df_store[col].dt.strftime("%Y-%m-%d")
    clean_df_store["_ingested_at"] = run_timestamp
    clean_df_store.to_sql("raw_sales_register", conn, if_exists="replace", index=False)

    n_demand = store(demand_lines, "demand_lines")
    n_pattern = store(pattern_classification, "part_demand_patterns")
    n_customer = store(customer_profile, "customer_profiles")
    n_py = store(part_yearly_trend, "part_yearly_trend")
    n_cy = store(customer_yearly_trend, "customer_yearly_trend")
    n_forecast = store(forecasts, "part_forecasts")

    monthly_store = monthly_demand.copy()
    monthly_store["month"] = monthly_store["month"].astype(str)
    monthly_store["_computed_at"] = run_timestamp
    monthly_store.to_sql("part_monthly_demand", conn, if_exists="replace", index=False)

    n_breakdown = store(customer_breakdown, "part_customer_breakdown")
    n_conc = store(concentration, "part_customer_concentration")
    n_pi = store(part_intelligence, "part_intelligence_summary")
    n_decomp = store(forecast_decomposition, "customer_forecast_decomposition")
    n_acc = store(accuracy, "part_forecast_accuracy")

    pivot_store = monthly_pivot.copy()
    pivot_store["_computed_at"] = run_timestamp
    pivot_store.to_sql("part_monthly_pivot", conn, if_exists="replace", index=False)

    n_tuned = store(tuned_params, "part_tuned_parameters")

    horizon_store = forecast_horizon.copy()
    horizon_store["_computed_at"] = run_timestamp
    horizon_store.to_sql("part_forecast_horizon", conn, if_exists="replace", index=False)

    horizon_adj_store = forecast_horizon_adjusted.copy()
    horizon_adj_store["_computed_at"] = run_timestamp
    horizon_adj_store.to_sql("part_forecast_horizon_adjusted", conn, if_exists="replace", index=False)

    qty_pivot_store = forecast_qty_pivot.copy()
    qty_pivot_store["_computed_at"] = run_timestamp
    qty_pivot_store.to_sql("part_forecast_qty_pivot", conn, if_exists="replace", index=False)

    occ_pivot_store = forecast_occurrence_pivot.copy()
    occ_pivot_store["_computed_at"] = run_timestamp
    occ_pivot_store.to_sql("part_forecast_occurrence_pivot", conn, if_exists="replace", index=False)

    n_proj = store(project_intelligence, "project_intelligence")
    n_risk = store(risk_alerts, "risk_alerts")

    forecast_log_store = forecast_log.copy()
    forecast_log_store.to_sql("part_forecast_log", conn, if_exists="replace", index=False)

    reconciliation_store = reconciliation.copy()
    reconciliation_store.to_sql("part_forecast_reconciliation", conn, if_exists="replace", index=False)

    reconciliation_summary_store = reconciliation_summary.copy()
    reconciliation_summary_store.to_sql("part_forecast_reconciliation_summary", conn, if_exists="replace", index=False)

    if item_master is not None:
        item_master_store = item_master.copy()
        item_master_store["_ingested_at"] = run_timestamp
        item_master_store.to_sql("item_master", conn, if_exists="replace", index=False)

    n_replen = store(replenishment, "replenishment_recommendations")
    customer_commitments.to_sql("customer_commitments", conn, if_exists="replace", index=False)
    conn.close()

    # 4b. Part master: Item Master + part transfer list side by side. Reuses the transfer list already in
    # the database when no new file is supplied, so refreshing either file alone updates part_master.
    print("\n--- Step 4b: Part Master ---")
    pm = run_part_transfer_step(db_path, part_transfer_filepath)
    if pm.get("built"):
        print(f"Part master: {pm['total_parts']:,} parts ({pm['in_both']:,} in both files, "
              f"{pm['item_master_only']:,} Item Master only, {pm['transfer_list_only']:,} transfer list only)")
    else:
        print("Part master not built: no part transfer list loaded yet.")

    print(f"Stored to {db_path}: raw_sales_register ({len(clean_df_store)} rows), demand_lines ({n_demand}), "
          f"part_demand_patterns ({n_pattern}), part_monthly_demand ({len(monthly_store)}), "
          f"part_forecasts ({n_forecast}), part_customer_breakdown ({n_breakdown}), "
          f"part_customer_concentration ({n_conc}), part_intelligence_summary ({n_pi}), "
          f"customer_forecast_decomposition ({n_decomp}), customer_profiles ({n_customer}), "
          f"project_intelligence ({n_proj}), risk_alerts ({n_risk}), replenishment_recommendations ({n_replen})")

    # 5. Output report
    print("\n--- Step 5: Report ---")
    report_path = output_dir / f"sciip_demand_lines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        demand_lines.to_excel(writer, sheet_name="Demand Lines", index=False)
        open_only = demand_lines[demand_lines["status"] == "OPEN"].sort_values("so_no")
        open_only.to_excel(writer, sheet_name="Open Lines Only", index=False)
        part_intelligence.to_excel(writer, sheet_name="Part Intelligence Summary", index=False)
        pattern_classification.to_excel(writer, sheet_name="Part Demand Patterns", index=False)
        customer_profile.to_excel(writer, sheet_name="Customer Profiles", index=False)
        part_yearly_trend.to_excel(writer, sheet_name="Part Yearly Trend", index=False)
        customer_yearly_trend.to_excel(writer, sheet_name="Customer Yearly Trend", index=False)
        forecasts.to_excel(writer, sheet_name="Part Forecasts", index=False)
        forecast_decomposition.to_excel(writer, sheet_name="Forecast by Customer", index=False)
        forecast_horizon_adjusted.to_excel(writer, sheet_name="Forecast (Customer-Adjusted)", index=False)
        customer_breakdown.to_excel(writer, sheet_name="Part Customer Breakdown", index=False)
        monthly_pivot.to_excel(writer, sheet_name="Part Monthly Reference", index=False)
        accuracy.to_excel(writer, sheet_name="Forecast Accuracy", index=False)
        tuned_params.to_excel(writer, sheet_name="Tuned Parameters", index=False)
        forecast_qty_pivot.to_excel(writer, sheet_name="6mo Forecast Qty", index=False)
        forecast_occurrence_pivot.to_excel(writer, sheet_name="6mo Forecast Occurrence", index=False)
        project_intelligence.to_excel(writer, sheet_name="Project Intelligence", index=False)
        risk_alerts.to_excel(writer, sheet_name="Risk Alerts", index=False)
        reconciliation_summary.to_excel(writer, sheet_name="Forecast Reconciliation", index=False)
        if item_master is not None:
            item_master.to_excel(writer, sheet_name="Item Master", index=False)
            demand_missing_df.to_excel(writer, sheet_name="IM Coverage - Demand Gaps", index=False)
            im_no_demand_df.to_excel(writer, sheet_name="IM Coverage - No Demand", index=False)
        part_master_df = load_part_master(db_path)
        if len(part_master_df):
            part_master_df.to_excel(writer, sheet_name="Part Master", index=False)
        replenishment.to_excel(writer, sheet_name="Replenishment", index=False)

        summary = pd.DataFrame({
            "metric": ["Total register rows ingested", "Rows excluded (data issues)",
                       f"Rows excluded (project filter: not in {PROJECT_ALLOWLIST})",
                       "Total demand lines reconstructed", "Closed lines", "Open lines",
                       "Lines needing review (over-shipped/adjusted)",
                       "Total original demand (units)", "Total shipped (units)", "Total still open (units)"],
            "value": [report.row_count_raw, report.row_count_raw - report.row_count_clean - excluded_by_project,
                      excluded_by_project, len(demand_lines), demand_lines["status"].eq("CLOSED").sum(),
                      demand_lines["status"].eq("OPEN").sum(), demand_lines["needs_review"].sum(),
                      demand_lines["original_qty"].sum(), demand_lines["shipped_qty"].sum(), demand_lines["open_qty"].sum()]
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)
    print(f"Report written to {report_path}")

    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] Pipeline complete.")
    return demand_lines, report_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <sales_register.xlsx> [item_master.xlsx] [customer_commitments.xlsx] [part_transfer.xlsx]")
        sys.exit(1)
    item_master_arg = sys.argv[2] if len(sys.argv) > 2 else None
    commitments_arg = sys.argv[3] if len(sys.argv) > 3 else None
    transfer_arg = sys.argv[4] if len(sys.argv) > 4 else None
    run_pipeline(sys.argv[1], item_master_filepath=item_master_arg, customer_commitments_filepath=commitments_arg,
                 part_transfer_filepath=transfer_arg)
