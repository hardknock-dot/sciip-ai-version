"""
part_transfer_integration.py

Glue between part_transfer.py and the rest of SCIIP (pipeline, refresh, dashboard).
It reads/writes only SQLite tables, so it needs no knowledge of other modules' internals.

Tables:
    part_transfer  - latest cleaned Etch unique list (one row per part)
    part_master    - Item Master + transfer list, side by side (what the dashboard reads)

Why `rebuild_part_master` exists: the two files are refreshed independently.
Uploading only a new Item Master rebuilds part_master with the transfer list
already stored, and vice versa. Nothing has to be re-uploaded.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

import part_transfer as pt
import ui_prefs

ITEM_MASTER_TABLE = "item_master"      # change if your pipeline names it differently
PART_TRANSFER_TABLE = "part_transfer"
PART_MASTER_TABLE = "part_master"
DEMAND_TABLE = "demand_lines"          # used only for the demand coverage check


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    return con.execute("select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone() is not None


def _to_sql_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include=["boolean", "string"]).columns:
        out[c] = out[c].astype(object).where(out[c].notna(), None)
    for c in out.select_dtypes(include=["datetime"]).columns:
        out[c] = out[c].dt.strftime("%Y-%m-%d")
    return out


def rebuild_part_master(db_path: str | Path, key: str = pt.KEY_COLUMN,
                        item_master_table: str = ITEM_MASTER_TABLE) -> dict[str, Any]:
    """(Re)build part_master from whatever is currently stored. Safe to call after
    either file is refreshed. Returns coverage stats."""
    with sqlite3.connect(str(db_path)) as con:
        if not _table_exists(con, PART_TRANSFER_TABLE):
            return {"built": False, "reason": "No transfer list stored yet."}
        tr = pd.read_sql(f'select * from "{PART_TRANSFER_TABLE}"', con, parse_dates=["transfer_date"])
        has_im = _table_exists(con, item_master_table)
        im = pd.read_sql(f'select * from "{item_master_table}"', con) if has_im else pd.DataFrame({key: []})
        demand = (pd.read_sql(f'select distinct part_code from "{DEMAND_TABLE}"', con)["part_code"]
                  if _table_exists(con, DEMAND_TABLE) else None)

        # SQLite gives plain object/float dtypes back; restore booleans for readability
        for c in pt.YESNO_COLS:
            if c in tr:
                tr[c] = tr[c].map({1: True, 0: False}).astype("boolean")
        for c in ("incomplete_row", "corp_list_price_cannot_be_priced"):
            if c in tr:
                tr[c] = tr[c].fillna(0).astype(bool)

        merged = pt.merge_with_item_master(im, tr, key=key)
        _to_sql_safe(merged).to_sql(PART_MASTER_TABLE, con, if_exists="replace", index=False, chunksize=5000)
        con.execute(f'create index if not exists idx_{PART_MASTER_TABLE}_key on "{PART_MASTER_TABLE}"({key})')

    stats = pt.coverage_report(merged, demand)
    stats.update(built=True, item_master_present=has_im)
    return stats


def run_part_transfer_step(db_path: str | Path, transfer_path: str | Path | None = None,
                           key: str = pt.KEY_COLUMN,
                           item_master_table: str = ITEM_MASTER_TABLE) -> dict[str, Any]:
    """Pipeline step. With a file: validate -> store snapshot -> rebuild part_master.
    Without a file: just rebuild part_master from stored tables (used when only the
    Item Master was refreshed). Raises ValueError on an invalid file so a staging
    refresh fails before anything goes live."""
    summary: dict[str, Any] = {"transfer_loaded": False}
    if transfer_path is not None:
        df, rep = pt.load_part_transfer(transfer_path)
        if rep["errors"]:
            raise ValueError("Part transfer list rejected: " + "; ".join(rep["errors"]))
        summary["transfer_rows"] = pt.store_part_transfer(df, db_path, PART_TRANSFER_TABLE)
        summary["transfer_loaded"] = True
        summary["warnings"] = rep["warnings"]
        summary["notes"] = rep["notes"]
    summary.update(rebuild_part_master(db_path, key=key, item_master_table=item_master_table))
    return summary


def load_part_master(db_path: str | Path) -> pd.DataFrame:
    """Dashboard loader. Empty frame if nothing has been built yet."""
    with sqlite3.connect(str(db_path)) as con:
        if not _table_exists(con, PART_MASTER_TABLE):
            return pd.DataFrame()
        df = pd.read_sql(f'select * from "{PART_MASTER_TABLE}"', con)
    # SQLite stores booleans as 0/1; restore real bools so filters and ~ work
    for c in ("in_item_master", "in_transfer_list",
              "xfer_incomplete_row", "xfer_corp_list_price_cannot_be_priced"):
        if c in df:
            df[c] = df[c].fillna(0).astype(bool)
    return df


_FLAG_COLS = ("in_item_master", "in_transfer_list",
              "xfer_incomplete_row", "xfer_corp_list_price_cannot_be_priced")

PREF_KEY = "part_master_columns"
FIXED_COLUMNS = ["part_code", "part_desc"]          # always shown
FLAG_OPTIONS = ["source"]                            # computed / status columns
# The recommended compact view. Anything missing from the data is skipped automatically.
DEFAULT_COLUMNS = ["source", "current_stock", "computed_available_qty", "open_vendor_qty",
                   "lead_time_days", "moq", "vendor", "standard_cost", "part_status_raw",
                   "xfer_status", "xfer_criticality", "xfer_transfer_date"]

_LABELS = {"part_code": "Part code", "part_desc": "Description", "source": "Source",
           "current_stock": "Stock",
           "computed_available_qty": "Available", "open_vendor_qty": "On order (vendor)",
           "customer_committed_qty": "Committed", "lead_time_days": "Lead time (d)", "moq": "MOQ",
           "purchase_multiple": "Multiple", "standard_cost": "Std cost",
           "part_status_raw": "SAP status", "obsolete_by_oem": "Obsolete (OEM)"}


def _restore_flags(df: pd.DataFrame) -> pd.DataFrame:
    """SQLite returns booleans as 0/1; restore real bools so filters and ~ behave."""
    out = df.drop(columns=["description_conflict"], errors="ignore")   # retired column from older builds
    for c in _FLAG_COLS:
        if c in out:
            out[c] = out[c].fillna(0).astype(bool)
    return out


def column_label(col: str) -> str:
    """Readable header: known names mapped, transfer-list columns prefixed 'Transfer'."""
    if col in _LABELS:
        return _LABELS[col]
    if col.startswith(pt.PREFIX):
        return "Transfer \u00b7 " + col[len(pt.PREFIX):].replace("_", " ").capitalize()
    return col.replace("_", " ").capitalize()


def column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Selectable columns, grouped. Part code / description are fixed and not listed."""
    skip = set(FIXED_COLUMNS) | set(_FLAG_COLS) | {"_label"}
    sap = [c for c in df.columns if c not in skip and not c.startswith(pt.PREFIX) and not c.startswith("_")]
    x = [c for c in df.columns if c.startswith(pt.PREFIX) and c not in skip]
    return {"flags": [c for c in FLAG_OPTIONS if c == "source" or c in df.columns],
            "sap": sap, "xfer": x}


def render_part_master_section(st, df: pd.DataFrame, key_prefix: str = "pm",
                               prefs_path=None) -> None:
    """Item Master tab body: Item Master and transfer-list data side by side, showing only the
    columns you choose (the Columns control at the top). Every other column stays in the data:
    select a row to see all its fields, or download the full table.

    `df` is the part_master table as loaded by the dashboard's cached load_table(), so a
    refresh (which clears that cache) updates this view automatically."""
    df = _restore_flags(df)
    if df.empty:
        st.info("No part master data yet - upload an Item Master and/or the part transfer list in Data Refresh.")
        return

    # ---- derived columns -------------------------------------------------------------------
    desc = df["part_desc"] if "part_desc" in df else pd.Series(pd.NA, index=df.index, dtype="object")
    if "xfer_description" in df:
        desc = desc.fillna(df["xfer_description"])
    full = df.copy()
    full["part_desc"] = desc
    full["source"] = ["Both" if a and b else "Item Master" if a else "Transfer list"
                      for a, b in zip(full["in_item_master"], full["in_transfer_list"])]
    groups = column_groups(full)
    available = set(groups["flags"]) | set(groups["sap"]) | set(groups["xfer"])

    # ---- coverage cards ----------------------------------------------------------------------
    counts = full["source"].value_counts()
    cards = [
        ("Total parts", len(full), "#06B6D4", "&bull; Master catalog"),
        ("In both files", int(counts.get("Both", 0)), "#10B981", "&bull; Matched in both"),
        ("Item Master only", int(counts.get("Item Master", 0)), "#6366F1", "&bull; SAP records only"),
        ("Transfer list only", int(counts.get("Transfer list", 0)), "#F59E0B", "&bull; Transfer list only")
    ]
    cols = st.columns(4)
    for col, (label, val, accent, sub) in zip(cols, cards):
        col.markdown(
            f'<div class="scp-kpi" style="--kpi-accent:{accent};">'
            f'<div class="scp-kpi-label">{label}</div>'
            f'<div class="scp-kpi-value" style="font-size:24px;">{val:,}</div>'
            f'<div class="scp-kpi-sub">{sub}</div></div>',
            unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ---- column selection (saved view -> recommended default) ----------------------------------
    k_flag, k_sap, k_x = f"{key_prefix}_cols_flags", f"{key_prefix}_cols_sap", f"{key_prefix}_cols_xfer"

    def _split(chosen):
        chosen = [c for c in chosen if c in available]
        return ([c for c in chosen if c in groups["flags"]], [c for c in chosen if c in groups["sap"]],
                [c for c in chosen if c in groups["xfer"]])

    def _apply(chosen):
        st.session_state[k_flag], st.session_state[k_sap], st.session_state[k_x] = _split(chosen)

    if k_flag not in st.session_state:
        saved = ui_prefs.load(prefs_path).get(PREF_KEY)
        _apply(saved if isinstance(saved, list) and saved else DEFAULT_COLUMNS)

    top = st.columns([1.3, 4.2, 1.5])
    with top[0]:
        n_shown = len(FIXED_COLUMNS) + len(st.session_state[k_flag]) + len(st.session_state[k_sap]) + len(st.session_state[k_x])
        panel = st.popover(f"\u2699 Columns ({n_shown} shown)") if hasattr(st, "popover") \
            else st.expander(f"\u2699 Columns ({n_shown} shown)")
        with panel:
            st.caption("Choose what the table shows. Part code and description are always shown; "
                       "the order you pick is the order of the columns.")
            st.multiselect("Status", groups["flags"], key=k_flag, format_func=column_label)
            st.multiselect("Item Master (SAP)", groups["sap"], key=k_sap, format_func=column_label)
            st.multiselect("Transfer list", groups["xfer"], key=k_x, format_func=column_label)
            b1, b2 = st.columns(2)
            b1.button("Recommended view", key=f"{key_prefix}_reset", on_click=lambda: _apply(DEFAULT_COLUMNS))
            if b2.button("Save as my default", key=f"{key_prefix}_save"):
                chosen = list(st.session_state[k_flag]) + list(st.session_state[k_sap]) + list(st.session_state[k_x])
                st.caption("Saved." if ui_prefs.save({PREF_KEY: chosen}, prefs_path) else "Could not save the view.")
    with top[1]:
        choice = st.radio("Show", ["All", "In both files", "Item Master only", "Transfer list only",
                                   ], horizontal=True, key=f"{key_prefix}_filter",
                          label_visibility="collapsed")

    # ---- search + filter ---------------------------------------------------------------------
    view = full.copy()
    view["_label"] = view["part_code"].astype(str) + " - " + view["part_desc"].fillna("").astype(str).str.slice(0, 60)
    st.markdown('<div class="scp-section-label">Search / select multiple parts</div>', unsafe_allow_html=True)
    picked = st.multiselect("Parts", view["_label"].tolist(), label_visibility="collapsed",
                            placeholder="Type to search part code or description...", key=f"{key_prefix}_parts")
    if picked:
        view = view[view["_label"].isin(picked)]
    else:
        q = st.text_input("Or filter the full list", placeholder="Search part code or description...",
                          key=f"{key_prefix}_text")
        if q:
            view = view[view["part_code"].astype(str).str.contains(q, case=False, na=False)
                        | view["part_desc"].fillna("").astype(str).str.contains(q, case=False, na=False)]
    if choice == "In both files":
        view = view[view["source"] == "Both"]
    elif choice == "Item Master only":
        view = view[view["source"] == "Item Master"]
    elif choice == "Transfer list only":
        view = view[view["source"] == "Transfer list"]

    # ---- table: chosen columns only ------------------------------------------------------------
    show = FIXED_COLUMNS + list(st.session_state[k_flag]) + list(st.session_state[k_sap]) + list(st.session_state[k_x])
    show = [c for c in dict.fromkeys(show) if c in view.columns]
    with top[2]:
        st.download_button("\u21e9 All columns (CSV)", data=view.drop(columns=["_label"]).to_csv(index=False).encode("utf-8"),
                           file_name="part_master_full.csv", mime="text/csv", key=f"{key_prefix}_dl")
    st.caption(f"{len(view):,} part(s) \u00b7 {len(show)} of {len(full.columns) - 1} columns shown. "
               f"Select a row for every field.")
    table = view[show].reset_index(drop=True)
    event = st.dataframe(table, use_container_width=True, hide_index=True, height=480,
                         column_config={c: st.column_config.Column(column_label(c)) for c in show},
                         on_select="rerun", selection_mode="single-row", key=f"{key_prefix}_table")

    # ---- full record of the selected part (columns hidden from the table live here) ---------------
    try:
        rows = list(event.selection.rows)
    except Exception:
        rows = []
    if rows and rows[0] < len(table):
        code = table.iloc[rows[0]]["part_code"]
        rec = view[view["part_code"] == code].iloc[0]
        items = [(("Transfer list" if c.startswith(pt.PREFIX) else "Item Master" if c in groups["sap"] else "General"),
                  column_label(c), rec[c]) for c in full.columns
                 if c != "_label" and pd.notna(rec.get(c))]
        with st.expander(f"All fields \u00b7 {code}", expanded=True):
            st.dataframe(pd.DataFrame(items, columns=["Group", "Field", "Value"]).astype({"Value": str}),
                         use_container_width=True, hide_index=True)
