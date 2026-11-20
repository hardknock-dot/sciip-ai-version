"""
part_transfer.py

Ingestion of the "Etch unique list" (transfer-related part data) and its
side-by-side merge with the SAP Item Master, so the complete part master
holds the details of BOTH files.

Design:
  - Reads ONLY the configured sheet and ONLY the mapped columns (fast on
    large files; other sheets such as sales summaries are never touched).
  - Every transfer-list column is prefixed `xfer_` in the merged output, so
    nothing collides with, or silently overrides, an Item Master field.
  - SAP Item Master stays the source of truth for shared concepts
    (description, status, cost). The transfer-list value sits beside it; neither
    file silently overrides the other.
  - Status / criticality codes are carried as raw codes. No meaning is
    invented for them.
  - Pure functions + one storage function. No Streamlit, no global state.

Typical use:
    df_x, report = load_part_transfer("Etch_unique_list_4.xlsx")
    df_master = merge_with_item_master(df_item_master, df_x)
    store_part_transfer(df_x, "sciip.db")
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration (edit here if the file's headers change)
# --------------------------------------------------------------------------- #

TRANSFER_SHEET = "Sheet1"
KEY_COLUMN = "part_code"
PREFIX = "xfer_"

# raw header -> canonical name (before the xfer_ prefix is applied)
COLUMN_MAP: dict[str, str] = {
    "Part Number": KEY_COLUMN,
    "Description": "description",
    "Extended to Spares?": "extended_to_spares",
    "Criticality": "criticality",
    "Previously Shipped As Spare?": "previously_shipped_as_spare",
    "Date of Transfer to Skyline": "transfer_date",
    "Item group": "item_group",
    "STATUS": "status",
    "STANDARD COST": "standard_cost",
    "NA_LIST_PRICE": "na_list_price",
    "Lam 1 yr usage": "lam_1yr_usage",
    "Lam 5 years Usage": "lam_5yr_usage",
    "Our 1 yr usage": "our_1yr_usage",
    "Cost used for pricing": "cost_used_for_pricing",
    "Corp list price": "corp_list_price",
    "Optimal procurement": "optimal_procurement",
    "Re-ordering point": "reordering_point",
    "Ordering Qty": "ordering_qty",
    "Considered Price": "considered_price",
    "ECCN": "eccn",
    "HS code": "hs_code",
}

NUMERIC_COLS = [
    "standard_cost", "na_list_price", "lam_1yr_usage", "lam_5yr_usage",
    "our_1yr_usage", "cost_used_for_pricing", "corp_list_price",
    "optimal_procurement", "reordering_point", "ordering_qty", "considered_price",
]
YESNO_COLS = ["extended_to_spares", "previously_shipped_as_spare"]
CODE_COLS = ["criticality", "item_group", "status", "eccn", "hs_code"]

# Text placeholders that mean "no value"
NULL_TOKENS = {"", "-", "--", "---", "N/A", "NA", "NAN", "NONE"}
# Cells like these stay NULL in the number but are recorded as a flag
CANNOT_PRICE_TEXT = "cannot be priced"

REQUIRED_RAW = ["Part Number"]
MIN_EXPECTED_COLUMNS = 0.8  # fraction of mapped headers that must exist


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #

def _clean_key(s: pd.Series) -> pd.Series:
    """Part numbers come as text or bare integers; normalise to stripped upper text."""
    out = s.map(lambda v: "" if pd.isna(v) else (str(int(v)) if isinstance(v, (int, np.integer)) or
                                                  (isinstance(v, float) and v.is_integer()) else str(v)))
    return out.astype(str).str.strip().str.upper()


def _clean_code(s: pd.Series) -> pd.Series:
    """Text codes: strip, null placeholders and a bare 0 become NA."""
    def f(v: Any) -> Any:
        if pd.isna(v):
            return pd.NA
        t = str(int(v)) if isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer()) else str(v)
        t = t.strip()
        return pd.NA if t.upper() in NULL_TOKENS or t == "0" else t
    return s.map(f).astype("string")


def _clean_numeric(s: pd.Series) -> pd.Series:
    """'-' / '--' / blanks become NaN; '$1,234.50' style text is parsed."""
    if s.dtype.kind in "if":
        return s.astype(float)
    txt = s.astype("string").str.strip()
    txt = txt.mask(txt.str.upper().isin(NULL_TOKENS))
    txt = txt.str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(txt, errors="coerce")


def _clean_yesno(s: pd.Series) -> pd.Series:
    m = {"Y": True, "YES": True, "N": False, "NO": False}
    return s.astype("string").str.strip().str.upper().map(m).astype("boolean")


def load_part_transfer(
    path: str | Path, sheet: str = TRANSFER_SHEET
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, clean and validate the unique list.

    Returns (df, report). `report` carries stats, notes and errors; a
    non-empty `report["errors"]` means the file must not be used.
    """
    report: dict[str, Any] = {"errors": [], "warnings": [], "notes": [], "stats": {}}
    path = Path(path)

    try:
        header = pd.read_excel(path, sheet_name=sheet, nrows=0, engine="openpyxl")
    except ValueError:
        report["errors"].append(f"Sheet '{sheet}' not found in {path.name}.")
        return pd.DataFrame(), report
    except Exception as exc:  # unreadable / corrupt file
        report["errors"].append(f"Could not read {path.name}: {exc!r}")
        return pd.DataFrame(), report

    present = [c for c in COLUMN_MAP if c in header.columns]
    missing = [c for c in COLUMN_MAP if c not in header.columns]
    if any(c not in present for c in REQUIRED_RAW):
        report["errors"].append("Required column 'Part Number' is missing.")
        return pd.DataFrame(), report
    if len(present) / len(COLUMN_MAP) < MIN_EXPECTED_COLUMNS:
        report["errors"].append(f"Too many expected columns missing: {missing}")
        return pd.DataFrame(), report
    if missing:
        report["warnings"].append(f"Columns not found (left empty): {missing}")

    # usecols keeps a huge workbook fast: only mapped columns are parsed
    raw = pd.read_excel(path, sheet_name=sheet, usecols=present, engine="openpyxl")
    report["stats"]["raw_rows"] = len(raw)
    df = raw.rename(columns=COLUMN_MAP)
    for canon in COLUMN_MAP.values():
        if canon not in df.columns:
            df[canon] = pd.NA

    df[KEY_COLUMN] = _clean_key(df[KEY_COLUMN])
    blank = df[KEY_COLUMN] == ""
    report["stats"]["blank_key_rows_dropped"] = int(blank.sum())
    df = df.loc[~blank].copy()

    # raw text of price cells, to flag "Cannot be priced" before numeric coercion
    cannot = df["corp_list_price"].astype("string").str.strip().str.lower().eq(CANNOT_PRICE_TEXT)

    for c in NUMERIC_COLS:
        df[c] = _clean_numeric(df[c])
    for c in YESNO_COLS:
        df[c] = _clean_yesno(df[c])
    for c in CODE_COLS:
        df[c] = _clean_code(df[c])
    df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    df["description"] = df["description"].astype("string").str.strip()
    df["corp_list_price_cannot_be_priced"] = cannot.reindex(df.index).fillna(False).astype(bool)

    # Duplicate keys: keep the last occurrence, say so
    dups = int(df.duplicated(KEY_COLUMN, keep="last").sum())
    if dups:
        df = df.drop_duplicates(KEY_COLUMN, keep="last")
        report["warnings"].append(f"{dups} duplicate part number(s); last occurrence kept.")
    report["stats"]["duplicate_keys_dropped"] = dups

    # Rows that carry only a part number
    core = ["description", "item_group", "status", "transfer_date"]
    incomplete = df[core].isna().all(axis=1)
    df["incomplete_row"] = incomplete
    if incomplete.any():
        report["notes"].append(f"{int(incomplete.sum())} row(s) have a part number but no other data.")

    for c in ("standard_cost", "corp_list_price", "considered_price"):
        neg = int((df[c] < 0).sum())
        if neg:
            report["warnings"].append(f"{neg} negative value(s) in {c}.")
    n_cp = int(df["corp_list_price_cannot_be_priced"].sum())
    if n_cp:
        report["notes"].append(f"{n_cp} part(s) marked 'Cannot be priced' in Corp list price.")

    df = df.sort_values(KEY_COLUMN).reset_index(drop=True)
    report["stats"].update(
        rows=len(df),
        transfer_dates=sorted({d.date().isoformat() for d in df["transfer_date"].dropna().unique()}),
        status_codes=df["status"].value_counts(dropna=True).to_dict(),
    )
    if len(df) == 0:
        report["errors"].append("No usable rows after cleaning.")

    ordered = [KEY_COLUMN] + [c for c in df.columns if c != KEY_COLUMN]
    return df[ordered], report


# --------------------------------------------------------------------------- #
# Merge with SAP Item Master
# --------------------------------------------------------------------------- #

def merge_with_item_master(
    item_master: pd.DataFrame,
    transfer: pd.DataFrame,
    key: str = KEY_COLUMN,
) -> pd.DataFrame:
    """Side-by-side complete part master (outer join on the part code).

    Item Master columns are untouched. Transfer columns are added with the
    `xfer_` prefix. Flags show where each part exists. Neither file overrides
    the other: both values are kept side by side.
    """
    if key not in item_master.columns:
        raise KeyError(f"Item Master has no '{key}' column; pass key=<its part code column>.")

    im = item_master.copy()
    im["_k"] = _clean_key(im[key])
    tr = transfer.rename(columns={c: f"{PREFIX}{c}" for c in transfer.columns if c != KEY_COLUMN})
    tr = tr.rename(columns={KEY_COLUMN: "_k"})

    # Drop any stale xfer_ columns already on the item master (re-merge safe)
    im = im.drop(columns=[c for c in im.columns if c.startswith(PREFIX)], errors="ignore")

    out = im.merge(tr, on="_k", how="outer", indicator=True)
    out["in_item_master"] = out["_merge"].isin(["left_only", "both"])
    out["in_transfer_list"] = out["_merge"].isin(["right_only", "both"])
    # one normalised part code for every row (also fills transfer-only parts)
    out[key] = out["_k"]
    out = out.drop(columns=["_k", "_merge"])

    # identity columns first, then item master, then transfer block
    xfer_cols = [c for c in out.columns if c.startswith(PREFIX)]
    flags = ["in_item_master", "in_transfer_list"]
    base = [c for c in out.columns if c not in xfer_cols and c not in flags and c != key]
    return out[[key] + flags + base + xfer_cols]


def coverage_report(merged: pd.DataFrame, demand_parts: pd.Series | None = None) -> dict[str, int]:
    """Counts for the refresh panel: overlap between the two files, and
    (optionally) how many parts with real demand are missing from each."""
    rep = {
        "total_parts": len(merged),
        "in_both": int((merged["in_item_master"] & merged["in_transfer_list"]).sum()),
        "item_master_only": int((merged["in_item_master"] & ~merged["in_transfer_list"]).sum()),
        "transfer_list_only": int((~merged["in_item_master"] & merged["in_transfer_list"]).sum()),
    }
    if demand_parts is not None:
        d = set(_clean_key(pd.Series(pd.Series(demand_parts).dropna().unique())))
        have_x = set(merged.loc[merged["in_transfer_list"], KEY_COLUMN].astype(str))
        have_i = set(merged.loc[merged["in_item_master"], KEY_COLUMN].astype(str))
        rep["demand_parts"] = len(d)
        rep["demand_parts_missing_from_transfer_list"] = len(d - have_x)
        rep["demand_parts_missing_from_item_master"] = len(d - have_i)
    return rep


# --------------------------------------------------------------------------- #
# Storage + refresh-panel validation
# --------------------------------------------------------------------------- #

def store_part_transfer(df: pd.DataFrame, db_path: str | Path, table: str = "part_transfer") -> int:
    """Replace the table with this snapshot (a refresh is always a full snapshot)."""
    out = df.copy()
    for c in out.select_dtypes(include=["boolean", "string"]).columns:
        out[c] = out[c].astype(object).where(out[c].notna(), None)
    out["transfer_date"] = out["transfer_date"].dt.strftime("%Y-%m-%d")
    with sqlite3.connect(str(db_path)) as con:
        out.to_sql(table, con, if_exists="replace", index=False, chunksize=5000)
        con.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_key ON "{table}"({KEY_COLUMN})')
    return len(out)


def validate_part_transfer_file(path: str | Path) -> dict[str, Any]:
    """Validation result shaped for the refresh panel:
    {'ok': bool, 'errors': [...], 'warnings': [...], 'notes': [...], 'stats': {...}}"""
    _, rep = load_part_transfer(path)
    rep["ok"] = not rep["errors"]
    return rep
