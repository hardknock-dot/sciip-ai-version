"""
SCIIP - Demand Intelligence & Customer Intelligence
Built on top of the Transaction State Engine's demand_lines output.

Demand pattern classification uses the Syntetos-Boylan method:
    ADI (Average Demand Interval) = how many periods, on average, between
        periods that actually had demand.
    CV^2 (squared coefficient of variation) = how much the ORDER QUANTITY
        varies, in periods that did have demand.

    ADI < 1.32  & CV^2 < 0.49  -> SMOOTH       (regular timing, regular qty)
    ADI >= 1.32 & CV^2 < 0.49  -> INTERMITTENT (irregular timing, regular qty)
    ADI < 1.32  & CV^2 >= 0.49 -> ERRATIC      (regular timing, irregular qty)
    ADI >= 1.32 & CV^2 >= 0.49 -> LUMPY        (irregular timing AND qty)
"""
import pandas as pd
import numpy as np


def build_part_monthly_demand(demand_lines: pd.DataFrame, as_of_month: pd.Period = None) -> pd.DataFrame:
    """
    One row per (project, part_code, month) with total original_qty ordered.
    Includes zero-demand months for every project+part group, extended from
    each part's own FIRST order through a shared as_of_month (default: the
    latest month present anywhere in the register) - NOT just to that part's
    own last order, so a part that's genuinely gone quiet shows trailing
    zero-demand months instead of just disappearing from the data.
    """
    df = demand_lines.dropna(subset=["so_date", "part_code", "project"]).copy()
    df["month"] = df["so_date"].dt.to_period("M")

    if as_of_month is None:
        as_of_month = df["month"].max()

    monthly = (
        df.groupby(["project", "part_code", "month"])
        .agg(qty=("original_qty", "sum"), order_lines=("line_id", "count"))
        .reset_index()
    )

    filled = []
    for (project, part_code), g in monthly.groupby(["project", "part_code"]):
        full_range = pd.period_range(g["month"].min(), as_of_month, freq="M")
        g2 = g.set_index("month")[["qty", "order_lines"]].reindex(full_range, fill_value=0)
        g2["project"] = project
        g2["part_code"] = part_code
        g2.index.name = "month"
        filled.append(g2.reset_index())

    return pd.concat(filled, ignore_index=True) if filled else pd.DataFrame()


def classify_demand_patterns(monthly_demand: pd.DataFrame) -> pd.DataFrame:
    """One row per (project, part_code): ADI, CV^2, classification, stats."""
    records = []
    for (project, part_code), g in monthly_demand.groupby(["project", "part_code"]):
        total_periods = len(g)
        nonzero = g[g["qty"] > 0]
        n_nonzero = len(nonzero)
        if n_nonzero == 0:
            continue

        adi = total_periods / n_nonzero
        mean_qty = nonzero["qty"].mean()
        std_qty = nonzero["qty"].std(ddof=0) if n_nonzero > 1 else 0.0
        cv2 = (std_qty / mean_qty) ** 2 if mean_qty > 0 else 0.0

        if adi < 1.32 and cv2 < 0.49:
            pattern = "Smooth"
        elif adi >= 1.32 and cv2 < 0.49:
            pattern = "Intermittent"
        elif adi < 1.32 and cv2 >= 0.49:
            pattern = "Erratic"
        else:
            pattern = "Lumpy"

        records.append({
            "project": project, "part_code": part_code, "active_months": total_periods,
            "months_with_demand": n_nonzero, "adi": round(adi, 2), "cv2": round(cv2, 2),
            "avg_monthly_qty_when_ordered": round(mean_qty, 2), "total_qty": g["qty"].sum(),
            "demand_pattern": pattern,
        })

    return pd.DataFrame(records).sort_values(["project", "total_qty"], ascending=[True, False])


def build_part_yearly_trend(demand_lines: pd.DataFrame) -> pd.DataFrame:
    """Pivot: one row per (project, part_code), one column per calendar year, values = qty ordered."""
    df = demand_lines.dropna(subset=["so_date", "part_code", "project"]).copy()
    df["year"] = df["so_date"].dt.year
    pivot = df.pivot_table(index=["project", "part_code"], columns="year", values="original_qty", aggfunc="sum", fill_value=0)
    pivot.columns = [str(c) for c in pivot.columns]
    pivot = pivot.reset_index()
    pivot["total_all_years"] = pivot.drop(columns=["project", "part_code"]).sum(axis=1)
    return pivot.sort_values("total_all_years", ascending=False)


def build_customer_profile(demand_lines: pd.DataFrame, as_of_date: pd.Timestamp = None) -> pd.DataFrame:
    """One row per customer: order behavior, revenue, recency, breadth."""
    if as_of_date is None:
        as_of_date = demand_lines["so_date"].max()

    df = demand_lines.dropna(subset=["cust_name", "so_date"]).copy()
    records = []
    for cust, g in df.groupby("cust_name"):
        order_dates = g["so_date"].sort_values().unique()
        n_orders = len(g)
        total_qty = g["original_qty"].sum()
        total_revenue = g["order_amt"].sum()
        first_order = g["so_date"].min()
        last_order = g["so_date"].max()
        recency_days = (as_of_date - last_order).days
        tenure_days = (last_order - first_order).days

        if len(order_dates) > 1:
            intervals = np.diff(order_dates).astype("timedelta64[D]").astype(int)
            avg_order_interval_days = float(np.mean(intervals))
        else:
            avg_order_interval_days = None

        open_lines = g[g["status"] == "OPEN"]

        records.append({
            "cust_name": cust, "total_demand_lines": n_orders, "distinct_parts": g["part_code"].nunique(),
            "distinct_projects": g["project"].nunique(), "total_qty_ordered": total_qty,
            "total_revenue": round(total_revenue, 2), "avg_order_value": round(total_revenue / n_orders, 2) if n_orders else 0,
            "first_order_date": first_order, "last_order_date": last_order, "recency_days": recency_days,
            "tenure_days": tenure_days,
            "avg_order_interval_days": round(avg_order_interval_days, 1) if avg_order_interval_days else None,
            "currently_open_lines": len(open_lines), "currently_open_qty": open_lines["open_qty"].sum(),
        })

    return pd.DataFrame(records).sort_values("total_revenue", ascending=False)


def build_customer_yearly_trend(demand_lines: pd.DataFrame) -> pd.DataFrame:
    """Pivot: one row per (customer, project), one column per calendar year, values = revenue."""
    df = demand_lines.dropna(subset=["so_date", "cust_name", "project"]).copy()
    df["year"] = df["so_date"].dt.year
    pivot = df.pivot_table(index=["cust_name", "project"], columns="year", values="order_amt", aggfunc="sum", fill_value=0)
    pivot.columns = [str(c) for c in pivot.columns]
    pivot = pivot.reset_index()
    pivot["total_all_years"] = pivot.drop(columns=["cust_name", "project"]).sum(axis=1)
    return pivot.sort_values("total_all_years", ascending=False)


def build_part_customer_breakdown(demand_lines: pd.DataFrame) -> pd.DataFrame:
    """One row per (project, part_code, cust_name): that customer's contribution to this part's demand."""
    df = demand_lines.dropna(subset=["so_date", "part_code", "project", "cust_name"]).copy()
    records = []
    for (project, part_code), part_group in df.groupby(["project", "part_code"]):
        part_total_qty = part_group["original_qty"].sum()
        for cust_name, g in part_group.groupby("cust_name"):
            order_dates = g["so_date"].sort_values().unique()
            if len(order_dates) > 1:
                intervals = pd.Series(order_dates).diff().dt.days.dropna()
                avg_interval = round(intervals.mean(), 1)
            else:
                avg_interval = None
            records.append({
                "project": project, "part_code": part_code, "cust_name": cust_name,
                "cust_qty": g["original_qty"].sum(),
                "pct_of_part_demand": round(100 * g["original_qty"].sum() / part_total_qty, 1) if part_total_qty else 0,
                "order_count": len(g), "first_order_date": g["so_date"].min(), "last_order_date": g["so_date"].max(),
                "avg_order_interval_days": avg_interval,
            })
    return pd.DataFrame(records)


def build_part_customer_concentration(customer_breakdown: pd.DataFrame) -> pd.DataFrame:
    """One row per (project, part_code): top customer and concentration."""
    records = []
    for (project, part_code), g in customer_breakdown.groupby(["project", "part_code"]):
        g_sorted = g.sort_values("cust_qty", ascending=False)
        top = g_sorted.iloc[0]
        records.append({
            "project": project, "part_code": part_code, "num_customers": len(g), "top_customer": top["cust_name"],
            "top_customer_pct_of_demand": top["pct_of_part_demand"],
            "top_customer_avg_order_interval_days": top["avg_order_interval_days"],
            "top_customer_last_order_date": top["last_order_date"],
        })
    return pd.DataFrame(records)


def build_part_intelligence_summary(pattern_classification: pd.DataFrame, forecasts: pd.DataFrame,
                                      concentration: pd.DataFrame) -> pd.DataFrame:
    """ONE consolidated, dashboard-ready table: classification + forecast + customer concentration per part."""
    merged = pattern_classification.merge(forecasts, on=["project", "part_code"], suffixes=("", "_fc"))
    merged = merged.merge(concentration, on=["project", "part_code"], how="left")
    forecast_horizon_col = [c for c in merged.columns if c.startswith("forecast_next_")][0]
    keep_cols = [
        "project", "part_code", "demand_pattern", "adi", "cv2", "total_qty", "months_with_demand", "active_months",
        "forecast_method", "occurrence_probability_this_month", "expected_qty_if_it_occurs",
        "predicted_to_occur_this_month", "forecast_monthly_qty", forecast_horizon_col,
        "num_customers", "top_customer", "top_customer_pct_of_demand",
        "top_customer_avg_order_interval_days", "top_customer_last_order_date",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    return merged[keep_cols].sort_values("total_qty", ascending=False)


def add_relationship_health(customer_breakdown: pd.DataFrame, as_of_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Adds a status per (project, part_code, cust_name) row: is this specific
    customer's ordering of this specific part on track, or has it gone
    quiet relative to THEIR OWN historical cadence for that part?
    """
    df = customer_breakdown.copy()
    if as_of_date is None:
        as_of_date = df["last_order_date"].max()
    df["days_since_last_order"] = (as_of_date - df["last_order_date"]).dt.days

    def status(row):
        if row["order_count"] < 2 or pd.isna(row["avg_order_interval_days"]):
            return "New / not enough history"
        ratio = row["days_since_last_order"] / row["avg_order_interval_days"]
        if ratio <= 1.25:
            return "Active"
        elif ratio <= 2.0:
            return "Due soon"
        else:
            return "Overdue / went quiet"

    df["relationship_status"] = df.apply(status, axis=1)
    return df


def build_project_intelligence(demand_lines: pd.DataFrame, pattern_classification: pd.DataFrame,
                                 forecasts: pd.DataFrame, customer_breakdown: pd.DataFrame) -> pd.DataFrame:
    """One row per project: demand, revenue, growth, concentration, pattern mix, risk count."""
    df = demand_lines.dropna(subset=["project", "so_date"]).copy()
    df["year"] = df["so_date"].dt.year
    records = []

    for project, g in df.groupby("project"):
        total_qty = g["original_qty"].sum()
        total_revenue = g["order_amt"].sum()
        num_customers = g["cust_name"].nunique()
        num_parts = g["part_code"].nunique()

        by_year = g.groupby("year")["order_amt"].sum().sort_index()
        if len(by_year) >= 2:
            prev, curr = by_year.iloc[-2], by_year.iloc[-1]
            yoy_growth_pct = round(100 * (curr - prev) / prev, 1) if prev > 0 else None
        else:
            yoy_growth_pct = None

        by_part_revenue = g.groupby("part_code")["order_amt"].sum().sort_values(ascending=False)
        top_part_pct_of_revenue = round(100 * by_part_revenue.iloc[0] / total_revenue, 1) if total_revenue > 0 and len(by_part_revenue) else None

        proj_patterns = pattern_classification[pattern_classification["project"] == project]
        pattern_mix = proj_patterns["demand_pattern"].value_counts().to_dict()

        proj_forecast = forecasts[forecasts["project"] == project]
        forecast_next_month_total = round(proj_forecast["forecast_monthly_qty"].sum(), 2) if len(proj_forecast) else None

        proj_breakdown = customer_breakdown[customer_breakdown["project"] == project]
        num_overdue = (proj_breakdown["relationship_status"] == "Overdue / went quiet").sum() if len(proj_breakdown) else 0

        records.append({
            "project": project, "total_qty": total_qty, "total_revenue": round(total_revenue, 2),
            "num_customers": num_customers, "num_parts": num_parts, "yoy_growth_pct": yoy_growth_pct,
            "top_part_pct_of_revenue": top_part_pct_of_revenue,
            "smooth_parts": pattern_mix.get("Smooth", 0), "intermittent_parts": pattern_mix.get("Intermittent", 0),
            "erratic_parts": pattern_mix.get("Erratic", 0), "lumpy_parts": pattern_mix.get("Lumpy", 0),
            "forecast_next_month_total": forecast_next_month_total, "overdue_relationships_count": num_overdue,
        })

    return pd.DataFrame(records).sort_values("total_revenue", ascending=False)


def compute_trend_direction(yearly_df: pd.DataFrame) -> pd.DataFrame:
    """Adds trend_direction (Growing/Stable/Declining/New) and yoy_change_pct comparing the last two years."""
    df = yearly_df.copy()
    year_cols = sorted([c for c in df.columns if str(c).isdigit()])
    if len(year_cols) < 2:
        df["trend_direction"] = "New / not enough history"
        df["yoy_change_pct"] = None
        return df
    prev_col, curr_col = year_cols[-2], year_cols[-1]

    def calc(row):
        prev, curr = row[prev_col], row[curr_col]
        if prev == 0 and curr == 0:
            return "New / not enough history", None
        if prev == 0:
            return "Growing", None
        pct = round(100 * (curr - prev) / prev, 1)
        if pct > 10:
            return "Growing", pct
        elif pct < -10:
            return "Declining", pct
        return "Stable", pct

    results = df.apply(calc, axis=1)
    df["trend_direction"] = results.apply(lambda x: x[0])
    df["yoy_change_pct"] = results.apply(lambda x: x[1])
    return df


def build_risk_alerts(demand_lines: pd.DataFrame, concentration: pd.DataFrame, part_yearly_trend: pd.DataFrame,
                        accuracy: pd.DataFrame, tuned_params: pd.DataFrame, monthly_demand: pd.DataFrame,
                        lead_time_threshold_days: float = 120, concentration_threshold_pct: float = 70,
                        low_accuracy_threshold_pct: float = 55) -> pd.DataFrame:
    """The standardized df_risk_alerts layer: rules applied to already-proven tables, nothing new computed."""
    records = []

    lt = demand_lines.dropna(subset=["project", "part_code"]).groupby(["project", "part_code"]).agg(
        avg_actual_lead_time=("actual_days_to_close", "mean"), avg_quoted_lead_time=("quoted_lead_time_days", "mean"),
    ).reset_index()
    for _, r in lt.iterrows():
        lead = r["avg_actual_lead_time"] if pd.notna(r["avg_actual_lead_time"]) else r["avg_quoted_lead_time"]
        if pd.notna(lead) and lead > lead_time_threshold_days:
            records.append({"alert_type": "Long lead-time risk", "severity": "High" if lead > lead_time_threshold_days * 1.5 else "Medium",
                             "project": r["project"], "part_code": r["part_code"], "cust_name": None,
                             "message": f"Average lead time is {lead:.0f} days, exceeding the {lead_time_threshold_days}-day threshold.",
                             "metric_value": round(lead, 1)})

    for _, r in concentration.iterrows():
        if pd.notna(r["top_customer_pct_of_demand"]) and r["top_customer_pct_of_demand"] > concentration_threshold_pct:
            records.append({"alert_type": "Customer concentration risk", "severity": "High" if r["top_customer_pct_of_demand"] > 85 else "Medium",
                             "project": r["project"], "part_code": r["part_code"], "cust_name": r["top_customer"],
                             "message": f"{r['top_customer']} accounts for {r['top_customer_pct_of_demand']:.0f}% of this part's demand.",
                             "metric_value": r["top_customer_pct_of_demand"]})

    py_trend = compute_trend_direction(part_yearly_trend)
    for _, r in py_trend.iterrows():
        if r["trend_direction"] == "Declining":
            records.append({"alert_type": "Declining demand", "severity": "Medium", "project": r["project"], "part_code": r["part_code"],
                             "cust_name": None, "message": f"Demand declined {r['yoy_change_pct']:.0f}% year over year.", "metric_value": r["yoy_change_pct"]})
        elif r["trend_direction"] == "Growing":
            records.append({"alert_type": "Increasing demand", "severity": "Opportunity", "project": r["project"], "part_code": r["part_code"],
                             "cust_name": None, "message": f"Demand grew {r['yoy_change_pct']:.0f}% year over year.", "metric_value": r["yoy_change_pct"]})

    if len(tuned_params) > 0:
        low_conf = tuned_params[tuned_params["met_baseline"] == False]
        for _, r in low_conf.iterrows():
            records.append({"alert_type": "Low forecast confidence", "severity": "Medium", "project": r["project"], "part_code": r["part_code"],
                             "cust_name": None, "message": "Tuned forecast could not beat its own naive baseline - limited history.",
                             "metric_value": r.get("tuned_occurrence_hit_rate_pct")})
    if len(accuracy) > 0:
        low_acc = accuracy[accuracy["occurrence_hit_rate_pct"] < low_accuracy_threshold_pct]
        for _, r in low_acc.iterrows():
            records.append({"alert_type": "Low forecast confidence", "severity": "Medium", "project": r["project"], "part_code": r["part_code"],
                             "cust_name": None, "message": f"Backtested occurrence hit rate is only {r['occurrence_hit_rate_pct']:.0f}%.",
                             "metric_value": r["occurrence_hit_rate_pct"]})

    if len(monthly_demand) > 0:
        for (project, part_code), g in monthly_demand.groupby(["project", "part_code"]):
            last_3 = g.sort_values("month").tail(3)
            if len(last_3) == 3 and (last_3["qty"] == 0).all():
                records.append({"alert_type": "Dormant part", "severity": "Medium", "project": project, "part_code": part_code,
                                 "cust_name": None, "message": "No demand recorded in the last 3 months.", "metric_value": 0})

    review_rows = demand_lines[demand_lines["needs_review"] == True]
    if len(review_rows) > 0:
        for (project, part_code), g in review_rows.groupby(["project", "part_code"]):
            records.append({"alert_type": "Needs review (adjustment)", "severity": "Low", "project": project, "part_code": part_code,
                             "cust_name": None, "message": f"{len(g)} line(s) flagged for over-shipment/credit-note adjustment.", "metric_value": len(g)})

    if not records:
        return pd.DataFrame(columns=["alert_type", "severity", "project", "part_code", "cust_name", "message", "metric_value"])
    return pd.DataFrame(records)


def build_customer_forecast_decomposition(part_intelligence: pd.DataFrame, customer_breakdown: pd.DataFrame) -> pd.DataFrame:
    """
    Breaks each part's aggregate monthly forecast down by customer, using
    each customer's historical share of that part's total demand as the
    allocation basis. This explains WHO the forecasted qty is expected to
    come from - a transparent split of the existing forecast number, not a
    separate per-customer model. relationship_status is carried through so a
    customer who's gone quiet doesn't silently get assumed to keep
    contributing their historical share.
    """
    forecast_lookup = part_intelligence.set_index(["project", "part_code"])["forecast_monthly_qty"].to_dict()
    records = []
    for (project, part_code), g in customer_breakdown.groupby(["project", "part_code"]):
        forecast_monthly = forecast_lookup.get((project, part_code))
        if forecast_monthly is None:
            continue
        for _, r in g.iterrows():
            share = (r["pct_of_part_demand"] or 0) / 100.0
            contribution = round(forecast_monthly * share, 2)
            records.append({
                "project": project, "part_code": part_code, "cust_name": r["cust_name"],
                "historical_pct_of_demand": r["pct_of_part_demand"], "forecast_contribution": contribution,
                "relationship_status": r.get("relationship_status"), "days_since_last_order": r.get("days_since_last_order"),
            })
    if not records:
        return pd.DataFrame(columns=["project", "part_code", "cust_name", "historical_pct_of_demand",
                                       "forecast_contribution", "relationship_status", "days_since_last_order"])
    return pd.DataFrame(records).sort_values(["project", "part_code", "forecast_contribution"], ascending=[True, True, False])
