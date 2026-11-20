"""
SCIIP - Replenishment Intelligence
Consumes the already-proven Transaction Engine and Forecasting Engine
outputs (demand_lines, part_intelligence_summary, risk_alerts) plus a
current Item Master snapshot - does NOT recompute transaction state,
fulfillment, or forecasts. This is purely a decision layer on top.

Core question answered per part: what should we replenish, how much, and
when - with every factor behind that answer visible, never a black-box score.
"""
import math
import pandas as pd


def aggregate_part_level_demand(part_intelligence: pd.DataFrame, risk_alerts: pd.DataFrame) -> pd.DataFrame:
    """
    Stock is physical - one pile per part code, shared across every project
    that uses it. Demand has been tracked per (project, part_code) everywhere
    upstream (correctly, since patterns can differ by project) - this step
    aggregates back up to the part_code level, which is what a real
    replenishment decision has to be made against.
    """
    records = []
    for part_code, g in part_intelligence.groupby("part_code"):
        total_forecast_monthly = g["forecast_monthly_qty"].sum()
        projects = sorted(g["project"].unique().tolist())
        # worst-case (most cautious) pattern across projects using this part
        pattern_priority = {"Lumpy": 0, "Erratic": 1, "Intermittent": 2, "Smooth": 3}
        worst_pattern = min(g["demand_pattern"].tolist(), key=lambda p: pattern_priority.get(p, 4))
        top_customer_row = g.loc[g["top_customer_pct_of_demand"].idxmax()] if g["top_customer_pct_of_demand"].notna().any() else None

        part_alerts = risk_alerts[risk_alerts["part_code"] == part_code] if len(risk_alerts) else pd.DataFrame()
        has_declining = "Declining demand" in part_alerts["alert_type"].values if len(part_alerts) else False
        has_growing = "Increasing demand" in part_alerts["alert_type"].values if len(part_alerts) else False
        has_low_confidence = "Low forecast confidence" in part_alerts["alert_type"].values if len(part_alerts) else False
        has_long_lead = "Long lead-time risk" in part_alerts["alert_type"].values if len(part_alerts) else False
        has_concentration_risk = "Customer concentration risk" in part_alerts["alert_type"].values if len(part_alerts) else False

        records.append({
            "part_code": part_code,
            "projects": ", ".join(projects),
            "num_projects": len(projects),
            "total_forecast_monthly_qty": round(total_forecast_monthly, 2),
            "worst_case_demand_pattern": worst_pattern,
            "total_qty_all_projects": g["total_qty"].sum(),
            "top_customer_pct_of_demand": top_customer_row["top_customer_pct_of_demand"] if top_customer_row is not None else None,
            "top_customer": top_customer_row["top_customer"] if top_customer_row is not None else None,
            "trend_declining": has_declining,
            "trend_growing": has_growing,
            "low_forecast_confidence": has_low_confidence,
            "long_lead_time_flagged": has_long_lead,
            "customer_concentration_risk": has_concentration_risk,
        })
    return pd.DataFrame(records)


def _lead_time_from_demand_history(demand_lines: pd.DataFrame) -> pd.DataFrame:
    """Fallback lead time per part, from actual fulfillment history, for parts Item Master doesn't cover."""
    return (demand_lines.dropna(subset=["part_code"])
            .groupby("part_code")
            .agg(fallback_lead_time_days=("actual_days_to_close", "mean"),
                 fallback_quoted_lead_time_days=("quoted_lead_time_days", "mean"))
            .reset_index())


def build_replenishment_recommendations(
    part_intelligence: pd.DataFrame,
    demand_lines: pd.DataFrame,
    risk_alerts: pd.DataFrame,
    item_master: pd.DataFrame = None,
    safety_lead_multiplier: float = 1.3,
    default_lead_time_days: float = 90,
) -> pd.DataFrame:
    """
    One row per part_code with a full, explainable replenishment
    recommendation. item_master is optional - if not provided (or a part
    is missing from it), current_stock is treated as unknown and the
    recommendation is flagged accordingly rather than guessed at.
    """
    agg = aggregate_part_level_demand(part_intelligence, risk_alerts)
    lead_fallback = _lead_time_from_demand_history(demand_lines)
    agg = agg.merge(lead_fallback, on="part_code", how="left")

    if item_master is not None and len(item_master) > 0:
        im_cols = [c for c in ["part_code", "part_desc", "current_stock", "computed_available_qty",
                                 "customer_committed_qty", "open_vendor_qty", "standard_cost",
                                 "lead_time_days", "moq", "purchase_multiple", "vendor",
                                 "part_status_raw", "obsolete_by_oem", "unconfirmed_suspect_status",
                                 "alternate_part_code"] if c in item_master.columns]
        agg = agg.merge(item_master[im_cols].rename(columns={"lead_time_days": "item_master_lead_time_days"}),
                          on="part_code", how="left")
    else:
        agg["current_stock"] = None
        agg["computed_available_qty"] = None
        agg["customer_committed_qty"] = None
        agg["open_vendor_qty"] = None
        agg["item_master_lead_time_days"] = None
        agg["moq"] = None
        agg["purchase_multiple"] = None
        agg["vendor"] = None
        agg["part_status_raw"] = None
        agg["obsolete_by_oem"] = None
        agg["unconfirmed_suspect_status"] = None
        agg["alternate_part_code"] = None

    records = []
    for _, r in agg.iterrows():
        has_stock_data = pd.notna(r.get("current_stock"))
        is_obsolete_by_oem = bool(r.get("obsolete_by_oem")) if pd.notna(r.get("obsolete_by_oem")) else False
        is_unconfirmed_suspect = bool(r.get("unconfirmed_suspect_status")) if pd.notna(r.get("unconfirmed_suspect_status")) else False
        has_alternate = pd.notna(r.get("alternate_part_code"))

        lead_time_days = r.get("item_master_lead_time_days")
        if pd.isna(lead_time_days):
            lead_time_days = r.get("fallback_lead_time_days")
        if pd.isna(lead_time_days):
            lead_time_days = r.get("fallback_quoted_lead_time_days")
        if pd.isna(lead_time_days):
            lead_time_days = default_lead_time_days
        lead_time_months = lead_time_days / 30.0

        forecast_monthly = r["total_forecast_monthly_qty"] or 0.0
        current_stock = r["current_stock"] if has_stock_data else None
        # computed_available_qty = current_stock - customer_committed_qty + open_vendor_qty,
        # already computed once in item_master.py - using it here directly avoids
        # double-counting open_vendor_qty a second time
        effective_position = r.get("computed_available_qty") if has_stock_data else None

        if forecast_monthly <= 0:
            months_of_coverage = None
        elif has_stock_data and pd.notna(effective_position):
            months_of_coverage = round(effective_position / forecast_monthly, 1)
        else:
            months_of_coverage = None

        # --- Criticality: business impact if this part stocks out (0-4 point scale, fully explainable) ---
        criticality_points = 0
        criticality_reasons = []
        if pd.notna(r["top_customer_pct_of_demand"]) and r["top_customer_pct_of_demand"] > 50:
            criticality_points += 1
            criticality_reasons.append("concentrated on one important customer")
        if lead_time_days > 120:
            criticality_points += 1
            criticality_reasons.append(f"{lead_time_days:.0f}-day lead time")
        if r["worst_case_demand_pattern"] in ("Lumpy", "Erratic"):
            criticality_points += 1
            criticality_reasons.append("unpredictable demand pattern")
        if r["num_projects"] > 1:
            criticality_points += 1
            criticality_reasons.append(f"used across {r['num_projects']} projects")
        criticality = "High" if criticality_points >= 3 else "Medium" if criticality_points >= 1 else "Low"

        # --- Supply risk: likelihood of a stockout given forecast trust + lead time + variability ---
        supply_risk_points = 0
        if r["low_forecast_confidence"]:
            supply_risk_points += 1
        if lead_time_days > 120:
            supply_risk_points += 1
        if r["worst_case_demand_pattern"] == "Lumpy":
            supply_risk_points += 1
        supply_risk = "High" if supply_risk_points >= 2 else "Medium" if supply_risk_points >= 1 else "Low"

        # --- Recommended quantity ---
        target_coverage_months = lead_time_months * safety_lead_multiplier
        target_stock = forecast_monthly * target_coverage_months
        recommended_qty_raw = max(0.0, target_stock - (effective_position or 0)) if has_stock_data else None
        # Only the still-UNCONFIRMED status codes block a recommendation outright -
        # OEM-obsolete parts still need a quantity (to source via an alternate),
        # they're just flagged differently below, not zeroed out
        recommended_qty = None if is_unconfirmed_suspect else recommended_qty_raw
        moq = r.get("moq")
        purchase_multiple = r.get("purchase_multiple")
        if recommended_qty is not None and recommended_qty > 0:
            if pd.notna(moq) and moq > 0 and recommended_qty < moq:
                recommended_qty = moq
            if pd.notna(purchase_multiple) and purchase_multiple > 1:
                recommended_qty = math.ceil(recommended_qty / purchase_multiple) * purchase_multiple

        # --- Status + reasons ---
        reasons = []
        if is_unconfirmed_suspect:
            status = "Do Not Replenish (Unconfirmed Status - Please Review)"
        elif not has_stock_data:
            status = "Stock Unknown - Upload Item Master"
        elif forecast_monthly <= 0:
            status = "No Forecasted Demand"
        elif months_of_coverage is None:
            status = "Stock Unknown - Upload Item Master"
        elif is_obsolete_by_oem and months_of_coverage < target_coverage_months:
            status = "Obsolete by OEM - Alternate Source Needed"
            reasons.append("OEM discontinued this part (Part Status: OB)")
            reasons.append("no alternate part mapped yet" if not has_alternate else f"alternate part on file: {r.get('alternate_part_code')}")
        elif months_of_coverage < lead_time_months:
            status = "Replenish Now"
            reasons.append(f"stock covers only {months_of_coverage:.1f} of {lead_time_months:.1f} months needed to cover lead time")
        elif months_of_coverage < target_coverage_months:
            status = "Replenish Soon"
            reasons.append(f"stock covers {months_of_coverage:.1f} months, below the {target_coverage_months:.1f}-month safety target")
        else:
            status = "Adequate"

        if status in ("Replenish Now", "Replenish Soon", "Obsolete by OEM - Alternate Source Needed"):
            reasons.append(f"{lead_time_days:.0f}-day lead time")
            if r["trend_growing"]:
                reasons.append("increasing demand")
            if r["trend_declining"]:
                reasons.append("declining demand - review before ordering full amount")
            if criticality == "High":
                reasons.append("high criticality")
            if supply_risk == "High":
                reasons.append("high supply risk")

        reason_text = " + ".join(reasons) if reasons else "-"

        records.append({
            "part_code": r["part_code"],
            "part_desc": r.get("part_desc"),
            "projects": r["projects"],
            "current_stock": current_stock,
            "customer_committed_qty": r.get("customer_committed_qty"),
            "open_vendor_qty": r.get("open_vendor_qty"),
            "effective_available_position": effective_position,
            "lead_time_days": round(lead_time_days, 0),
            "lead_time_source": "Item Master" if pd.notna(r.get("item_master_lead_time_days")) else "Actual fulfillment history" if pd.notna(r.get("fallback_lead_time_days")) else "Default assumption",
            "forecast_monthly_qty": round(forecast_monthly, 2),
            "months_of_coverage": months_of_coverage,
            "demand_pattern": r["worst_case_demand_pattern"],
            "criticality": criticality,
            "criticality_reasons": ", ".join(criticality_reasons) if criticality_reasons else "-",
            "supply_risk": supply_risk,
            "top_customer": r.get("top_customer"),
            "top_customer_pct_of_demand": r.get("top_customer_pct_of_demand"),
            "vendor": r.get("vendor"),
            "moq": moq if pd.notna(moq) else None,
            "purchase_multiple": purchase_multiple if pd.notna(purchase_multiple) else None,
            "part_status_raw": r.get("part_status_raw"),
            "obsolete_by_oem": is_obsolete_by_oem,
            "alternate_part_code": r.get("alternate_part_code"),
            "unconfirmed_status_flag": is_unconfirmed_suspect,
            "status": status,
            "recommended_qty": round(recommended_qty, 0) if recommended_qty is not None else None,
            "reason": reason_text,
        })

    result = pd.DataFrame(records)
    status_priority = {"Replenish Now": 0, "Obsolete by OEM - Alternate Source Needed": 1, "Replenish Soon": 2,
                        "Stock Unknown - Upload Item Master": 3, "Adequate": 4, "No Forecasted Demand": 5,
                        "Do Not Replenish (Unconfirmed Status - Please Review)": 6}
    result["_sort"] = result["status"].map(status_priority)
    return result.sort_values("_sort").drop(columns="_sort")
