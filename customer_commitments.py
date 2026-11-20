"""
SCIIP - Customer-Provided Forecast Commitments
Lets a customer's own stated plan override our statistical estimate for
THEIR slice of a part's forecast, for a specific future month - without
touching any other customer's contribution or the underlying statistical
model. This is an adjustment layer on top of the proven forecast engine,
not a replacement for it.

Expected input Excel columns (flexible naming, mapped below):
    Project, Part Code, Customer, Month (YYYY-MM), Committed Qty
"""
import pandas as pd
from pathlib import Path

COLUMN_MAP = {
    "Project": "project", "Part Code": "part_code", "Part No.": "part_code", "Part No": "part_code",
    "Customer": "cust_name", "Customer Name": "cust_name",
    "Month": "month", "Period": "month",
    "Committed Qty": "committed_qty", "Committed Quantity": "committed_qty", "Forecast Qty": "committed_qty",
}
REQUIRED_COLS = ["project", "part_code", "cust_name", "month", "committed_qty"]


def load_customer_commitments(filepath: str) -> pd.DataFrame:
    """Reads a customer-provided forecast/commitment file and standardizes it."""
    path = Path(filepath)
    df = pd.read_excel(path, dtype=str) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path, dtype=str)

    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}. "
                          f"Expected something like: Project, Part Code, Customer, Month, Committed Qty")

    df["committed_qty"] = pd.to_numeric(df["committed_qty"], errors="coerce")
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.strftime("%Y-%m")
    bad_rows = df[df["committed_qty"].isna() | df["month"].isna()]
    if len(bad_rows) > 0:
        print(f"WARNING: {len(bad_rows)} commitment rows have an unparseable Month or Committed Qty - excluded")
        df = df.dropna(subset=["committed_qty", "month"])

    return df[REQUIRED_COLS].reset_index(drop=True)


def apply_customer_commitments(forecast_horizon: pd.DataFrame, forecast_decomposition: pd.DataFrame,
                                 customer_commitments: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts the part-level forecast wherever a customer has told us their own
    planned quantity for a specific future month. For that customer's slice
    only, their stated number REPLACES the statistical estimate (their
    historical share of the forecast); every other customer's contribution
    is left untouched - "swap this one customer's estimate for their real
    number", not a black-box reweighting of the whole forecast.
    """
    fh = forecast_horizon.copy()
    fh["adjusted_forecast_qty"] = fh["forecast_qty"]
    fh["adjustment_note"] = None

    if customer_commitments is None or len(customer_commitments) == 0:
        return fh

    stat_share = forecast_decomposition.set_index(["project", "part_code", "cust_name"])["forecast_contribution"].to_dict()

    for _, c in customer_commitments.iterrows():
        key = (c["project"], c["part_code"], c["cust_name"])
        statistical_estimate = stat_share.get(key, 0.0)
        delta = c["committed_qty"] - statistical_estimate

        mask = (fh["project"] == c["project"]) & (fh["part_code"] == c["part_code"]) & (fh["forecast_month"] == c["month"])
        if mask.sum() == 0:
            continue  # that month isn't in the current forecast horizon - nothing to adjust
        fh.loc[mask, "adjusted_forecast_qty"] = (fh.loc[mask, "adjusted_forecast_qty"] + delta).clip(lower=0)
        note = f"{c['cust_name']} provided {c['committed_qty']:.0f} (was est. {statistical_estimate:.1f})"
        fh.loc[mask, "adjustment_note"] = fh.loc[mask, "adjustment_note"].fillna("") + note + "; "

    return fh
