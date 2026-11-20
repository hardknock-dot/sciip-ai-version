"""
SCIIP - Item Master Ingestion & Normalization
Reads the real SAP-adjacent ITEM MASTER.xlsx export, maps it to standardized
SCIIP fields, cleans/validates it, and produces a normalized DataFrame the
Replenishment engine consumes. Nothing here recalculates Transaction State,
Demand Intelligence, or Forecasts - those stay exactly as already built.

Column mapping below reflects the ACTUAL file inspected (44 columns,
27,377 rows) - not a generic guess. Several fields required real judgment
calls, documented inline; anything genuinely ambiguous is carried through
raw and flagged in the validation report, never silently interpreted.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# --- Direct, unambiguous column mapping (verified against the real file) ---
COLUMN_MAP = {
    "Item No.": "part_code",
    "Item Description": "part_desc",
    "Item Grp": "item_group",              # fully populated (8 groups) - the REAL group field
    "Material Group": "material_group_raw", # 100% EMPTY in the real file - kept for completeness only
    "Part Status": "part_status_raw",       # cryptic codes (P/IA/NF/D/OB/...) - meaning NOT invented, see below
    "Lead Time": "lead_time_days",          # only 8.8% populated in the real file - fallback matters
    "In Stock": "current_stock",
    "Qty Ordered from Vendors": "open_vendor_qty",
    "Qty Ordered by Customers": "customer_committed_qty",
    "Minimum Order Quantity": "moq",
    "Quantity per Package (Purchasing)": "purchase_multiple",
    "Vendor": "vendor",
    "Project": "project_code",              # raw codes (EH001/INV001/LAM001/ONTO001/SP001) - NOT the
                                              # same taxonomy as Sales Register's project names; kept separate
    "Multi Project": "multi_project_code",   # NOT a Yes/No flag despite the name - see validation notes
    "Default Warehouse": "warehouse",
    "Last Purchase Date": "last_purchase_date",
    "Last Purchase Price": "last_purchase_price",
    "Standard Cost": "standard_cost_raw",    # mixed types incl. "$1,234.56" strings - cleaned below
    "Consumable": "_consumable_flag",
    "Non-Consumable": "_non_consumable_flag",
    "L": "unclear_flag_l",                   # meaning not documented in source file - carried through raw only
    "Z": "unclear_flag_z",                   # meaning not documented in source file - carried through raw only
}

# Six separate Yes/No project columns in the real file, rather than one field
PROJECT_FLAG_COLUMNS = ["inova 200", "inova 300m", "ONTO", "AXCELIS", "LAM DISTRIBUTION", "SP-203"]

NUMERIC_COLS = ["lead_time_days", "current_stock", "open_vendor_qty", "customer_committed_qty",
                 "moq", "purchase_multiple", "last_purchase_price"]
REQUIRED_COLS = ["part_code"]

# CONFIRMED by user: "OB" = Obsolete by OEM - the original manufacturer no
# longer supplies it, but the business still tries to establish alternate
# supply (a substitute part, or a different vendor) rather than simply
# writing the part off. This is NOT a hard "never replenish" situation.
CONFIRMED_OBSOLETE_BY_OEM_CODES = {"OB"}

# Still UNCONFIRMED - "D"/"DB" look obsolete/discontinued-ish by common
# convention, but the user hasn't confirmed this, so these stay conservative
# (blocked + flagged for review) rather than assumed.
UNCONFIRMED_SUSPECT_CODES = {"D", "DB"}

COST_OUTLIER_THRESHOLD = 50_000  # flag for review, not rejected - real capital parts can legitimately cost this much


class ItemMasterReport:
    def __init__(self):
        self.issues: list[str] = []
        self.row_count_raw = 0
        self.row_count_clean = 0

    def add(self, msg: str):
        self.issues.append(msg)

    def summary(self) -> str:
        lines = [f"Rows read: {self.row_count_raw}", f"Rows retained: {self.row_count_clean}",
                 f"Issues found: {len(self.issues)}"]
        lines += [f"  - {i}" for i in self.issues]
        return "\n".join(lines)


def _clean_cost(x):
    """Standard Cost is mixed-type in the real file: plain numbers, and
    strings like '$1,706,015.75'. Strip currency formatting, coerce to float."""
    if isinstance(x, str):
        return pd.to_numeric(x.replace("$", "").replace(",", "").strip(), errors="coerce")
    return x


def load_item_master(filepath: str) -> tuple[pd.DataFrame, ItemMasterReport]:
    report = ItemMasterReport()
    path = Path(filepath)

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    report.row_count_raw = len(df)
    df.columns = [c.strip() for c in df.columns]

    unmapped = [c for c in df.columns if c not in COLUMN_MAP and c not in PROJECT_FLAG_COLUMNS]
    if unmapped:
        report.add(f"Unmapped columns found (kept out of the normalized output): {unmapped}")

    project_flags_df = df[[c for c in PROJECT_FLAG_COLUMNS if c in df.columns]].copy()
    df = df.rename(columns=COLUMN_MAP)
    df = df.loc[:, ~df.columns.duplicated()]

    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns after mapping: {missing_required}. "
                          f"Check real column headers against item_master.py's COLUMN_MAP.")

    # Standardize types
    for col in df.columns:
        if col in df and df[col].dtype == object:
            df[col] = df[col].str.strip().replace({"": None, "nan": None, "NaN": None})

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "standard_cost_raw" in df.columns:
        df["standard_cost"] = df["standard_cost_raw"].apply(_clean_cost)
        outliers = df[df["standard_cost"] > COST_OUTLIER_THRESHOLD]
        if len(outliers) > 0:
            report.add(f"{len(outliers)} parts have Standard Cost above ${COST_OUTLIER_THRESHOLD:,} "
                        f"(up to ${outliers['standard_cost'].max():,.0f}) - flagged for review, not excluded")

    if "last_purchase_date" in df.columns:
        df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"], errors="coerce", format="mixed")

    # --- item_type: derived from two separate Yes/No columns, not one field ---
    # Real data has NO (Yes,Yes) rows, but a large "neither" group (~51%) that
    # is genuinely undetermined - never guessed at, always "Unspecified"
    if "_consumable_flag" in df.columns and "_non_consumable_flag" in df.columns:
        def item_type(row):
            cons = str(row.get("_consumable_flag", "")).strip().lower() == "yes"
            non_cons = str(row.get("_non_consumable_flag", "")).strip().lower() == "yes"
            if cons and not non_cons:
                return "Consumable"
            elif non_cons and not cons:
                return "Non-Consumable"
            elif cons and non_cons:
                return "Conflicting flags"  # doesn't occur in the real file, but handled rather than assumed
            return "Unspecified"
        df["item_type"] = df.apply(item_type, axis=1)
        df = df.drop(columns=["_consumable_flag", "_non_consumable_flag"])

    # --- item_master_project_tags: combine the six Yes/No project columns into one readable field ---
    if len(project_flags_df) == len(df):
        def tags(row):
            return ", ".join(c for c in PROJECT_FLAG_COLUMNS if c in row.index and str(row[c]).strip().lower() == "yes")
        df["item_master_project_tags"] = project_flags_df.apply(tags, axis=1)

    # --- Part Status: raw code preserved. "OB" = confirmed Obsolete by OEM
    # (still needs replenishment logic - via alternate sourcing, not a block).
    # "D"/"DB" remain an unconfirmed guess, handled conservatively.
    if "part_status_raw" in df.columns:
        df["obsolete_by_oem"] = df["part_status_raw"].isin(CONFIRMED_OBSOLETE_BY_OEM_CODES)
        df["unconfirmed_suspect_status"] = df["part_status_raw"].isin(UNCONFIRMED_SUSPECT_CODES)

    # Placeholder slot for alternate/substitute part mapping - not populated
    # yet (no data source for this exists), but the column exists now so
    # wiring in real alternate-part data later is a pure data fill, not a
    # code change to replenishment.py
    df["alternate_part_code"] = None

    # --- Computed available-to-promise: source file has NO "Available" column at all ---
    # Standard MRP formula (stock - committed + incoming), clearly labeled as
    # SCIIP-computed rather than sourced, since we're not inventing a column
    # that doesn't exist - we're applying a well-established, named formula.
    if all(c in df.columns for c in ["current_stock", "customer_committed_qty", "open_vendor_qty"]):
        df["computed_available_qty"] = (df["current_stock"].fillna(0) - df["customer_committed_qty"].fillna(0)
                                          + df["open_vendor_qty"].fillna(0))

    # --- Structural validation (report, never silently drop without saying so) ---
    null_codes = df["part_code"].isna().sum()
    if null_codes > 0:
        report.add(f"{null_codes} rows have a blank Item No. - excluded from the normalized output")
        df = df[df["part_code"].notna()]

    dupes = df[df.duplicated(subset=["part_code"], keep=False)]
    if len(dupes) > 0:
        report.add(f"{dupes['part_code'].nunique()} part codes appear on multiple rows - "
                    f"first occurrence kept, rest dropped")
        df = df.drop_duplicates(subset=["part_code"], keep="first")

    if "current_stock" in df.columns:
        neg_stock = (df["current_stock"] < 0).sum()
        if neg_stock > 0:
            report.add(f"{neg_stock} rows have negative In Stock - kept as-is, flagged for review "
                        f"(not zeroed out, since a negative could reflect a real backorder/adjustment convention)")

    if "lead_time_days" in df.columns:
        missing_lt = df["lead_time_days"].isna().sum()
        pct = 100 * missing_lt / len(df) if len(df) else 0
        report.add(f"{missing_lt} of {len(df)} rows ({pct:.0f}%) have no Lead Time - "
                    f"Replenishment will fall back to actual fulfillment history where available")

    if "material_group_raw" in df.columns and df["material_group_raw"].notna().sum() == 0:
        report.add("Material Group column is present but 100% empty in this file - not used; "
                    "Item Grp is the populated grouping field")

    if "multi_project_code" in df.columns and "project_code" in df.columns:
        both = df[df["project_code"].notna() & df["multi_project_code"].notna()]
        if len(both) > 0:
            mismatch = (both["project_code"] != both["multi_project_code"]).sum()
            report.add(f"'Multi Project' does not behave as a Yes/No flag - it holds project codes nearly "
                        f"identical to 'Project' ({len(both) - mismatch}/{len(both)} match exactly). "
                        f"{mismatch} rows differ - please confirm the intended meaning before relying on it")

    if "obsolete_by_oem" in df.columns:
        n_ob = df["obsolete_by_oem"].sum()
        if n_ob > 0:
            report.add(f"{n_ob} parts are confirmed Obsolete by OEM (Part Status 'OB') - these still get a "
                        f"replenishment quantity computed, flagged for ALTERNATE SOURCING rather than blocked outright")
    if "unconfirmed_suspect_status" in df.columns:
        n_unconfirmed = df["unconfirmed_suspect_status"].sum()
        if n_unconfirmed > 0:
            report.add(f"{n_unconfirmed} parts have Part Status 'D' or 'DB' - meaning still UNCONFIRMED, "
                        f"treated conservatively (blocked from replenishment) until confirmed")

    report.add("No 'Available' column exists in the source file - computed_available_qty is a SCIIP-computed "
                "estimate (Stock - Customer Committed + Open Vendor Orders), not a value read from the file")

    # Actually restrict to the normalized schema - columns not in COLUMN_MAP's
    # targets were reported as "excluded" above, so make that true, not just
    # unrenamed-and-still-present. Ordered for readability: identity ->
    # stock/coverage -> supply/purchasing -> cost -> project context ->
    # flagged/uncertain fields last, so a quick scan reads top-to-bottom.
    normalized_cols = [c for c in [
        "part_code", "part_desc", "item_group", "item_type",
        "current_stock", "customer_committed_qty", "open_vendor_qty", "computed_available_qty",
        "lead_time_days", "moq", "purchase_multiple", "vendor", "warehouse",
        "standard_cost", "last_purchase_price", "last_purchase_date",
        "project_code", "item_master_project_tags", "multi_project_code",
        "part_status_raw", "obsolete_by_oem", "unconfirmed_suspect_status", "alternate_part_code",
        "material_group_raw", "unclear_flag_l", "unclear_flag_z", "standard_cost_raw",
    ] if c in df.columns]
    df = df[normalized_cols]

    report.row_count_clean = len(df)
    return df.reset_index(drop=True), report


def check_demand_coverage(item_master: pd.DataFrame, demand_lines: pd.DataFrame) -> dict:
    """
    Cross-references Item Master against actual demand history:
    - parts with real demand history but missing from Item Master entirely
    - parts in Item Master with no demand history at all (could be genuinely
      unused, new, or a part-code mismatch between the two systems)
    Returns both lists plus counts - this needs BOTH real files to be
    meaningful, so it's a separate function called once both are loaded.
    """
    im_parts = set(item_master["part_code"].dropna().unique())
    demand_parts = set(demand_lines["part_code"].dropna().unique())

    demand_missing_from_item_master = sorted(demand_parts - im_parts)
    item_master_with_no_demand = sorted(im_parts - demand_parts)

    return {
        "demand_parts_missing_from_item_master": demand_missing_from_item_master,
        "demand_parts_missing_count": len(demand_missing_from_item_master),
        "item_master_parts_with_no_demand": item_master_with_no_demand,
        "item_master_parts_with_no_demand_count": len(item_master_with_no_demand),
        "item_master_total_parts": len(im_parts),
        "demand_total_parts": len(demand_parts),
    }
