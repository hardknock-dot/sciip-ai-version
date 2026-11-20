"""
SCIIP - Data Refresh Service
Backend for the dashboard's upload-and-refresh feature. It contains NO
business logic of its own - every calculation still happens in
pipeline.run_pipeline() and the modules it calls. This module only:

    1. validates an uploaded file (reusing the existing ingestion loaders)
    2. runs the existing pipeline against a STAGING copy of the database
    3. verifies the result, then atomically swaps staging -> live
    4. records a refresh history entry (success or failure)

Because processing happens on a staging copy, a failed or interrupted
refresh can never leave the live database half-updated: the dashboard keeps
showing the previous complete dataset until a new one is fully ready.
"""
import contextlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from ingest import load_sales_register, COLUMN_MAP as SALES_COLUMN_MAP, REQUIRED_COLS as REQUIRED_SALES_COLS
from item_master import load_item_master
from customer_commitments import load_customer_commitments
from part_transfer import load_part_transfer
from pipeline import run_pipeline, DB_PATH, OUTPUT_DIR, PROJECT_ALLOWLIST

DATA_DIR = DB_PATH.parent
REFRESH_LOG_PATH = DATA_DIR / "refresh_log.json"
LOCK_PATH = DATA_DIR / ".refresh.lock"
STALE_LOCK_SECONDS = 30 * 60
ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv"}
MAX_HISTORY = 50

# A refresh is only accepted if these live tables exist and are non-empty
CRITICAL_TABLES = ["demand_lines", "part_intelligence_summary", "customer_profiles", "part_monthly_demand"]


class RefreshBusy(Exception):
    """Raised when another refresh is already running."""


@dataclass
class ValidationResult:
    kind: str
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)   # informational notes, non-blocking
    attention: list = field(default_factory=list)  # decision-relevant warnings requiring acknowledgement
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# Saving uploads
# ---------------------------------------------------------------------------
def save_upload(file_bytes: bytes, original_name: str, kind: str) -> Path:
    """Writes uploaded bytes to a temp file, preserving the extension."""
    suffix = Path(original_name).suffix.lower()
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"sciip_{kind}_"))
    path = tmp_dir / f"{kind}{suffix}"
    path.write_bytes(file_bytes)
    return path


def _extension_error(path: Path):
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return f"Unsupported file type '{path.suffix}'. Please upload an Excel (.xlsx/.xls) or CSV file."
    if path.stat().st_size == 0:
        return "The uploaded file is empty."
    return None


# ---------------------------------------------------------------------------
# Validation (reuses the existing loaders - no duplicated parsing rules)
# ---------------------------------------------------------------------------
def _live_snapshot() -> dict:
    """What's currently live, used to sanity-check a new upload against it."""
    snap = {"max_so_date": None, "row_count": None}
    if not DB_PATH.exists():
        return snap
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT MAX(so_date), COUNT(*) FROM raw_sales_register").fetchone()
        conn.close()
        snap["max_so_date"] = pd.to_datetime(row[0]) if row and row[0] else None
        snap["row_count"] = row[1] if row else None
    except Exception:
        pass
    return snap


def _sap_headers(internal_names) -> list:
    """Translate internal field names back to the SAP header text users recognise."""
    out = []
    for name in internal_names:
        out.append(next((sap for sap, internal in SALES_COLUMN_MAP.items() if internal == name), name))
    return out


def validate_sales_register(path: Path) -> ValidationResult:
    res = ValidationResult(kind="Sales Register")
    err = _extension_error(path)
    if err:
        res.errors.append(err)
        return res

    try:
        df, report = load_sales_register(str(path))
    except ValueError as e:
        if "Missing required columns" in str(e):
            res.errors.append(
                f"This does not look like a SAP Sales Register export - required columns were not found "
                f"({', '.join(_sap_headers(REQUIRED_SALES_COLS))}). Make sure the column headers are in the "
                f"first row of the sheet and that you exported the Sales Register, not a different report.")
        else:
            res.errors.append("The file could not be opened as an Excel/CSV workbook. It may be corrupted, "
                              "password-protected, or have the wrong file extension.")
        return res
    except Exception as e:
        res.errors.append(f"The file could not be read as a Sales Register ({type(e).__name__}). "
                          f"It may be corrupted or in an unexpected format.")
        return res

    if len(df) == 0:
        res.errors.append("No usable rows were found after validation (all rows were missing SO No., Part Code or SO Qty).")
        return res

    for issue in report.issues:
        res.warnings.append(issue)

    in_scope = df[df["project"].isin(PROJECT_ALLOWLIST)] if "project" in df.columns else df.iloc[0:0]
    if len(in_scope) == 0:
        found = sorted(df["project"].dropna().unique().tolist())[:10] if "project" in df.columns else []
        res.errors.append(
            f"None of the rows belong to the projects in scope {PROJECT_ALLOWLIST}. "
            f"Projects found in this file: {found or 'none (Project column empty)'}.")
        return res

    so_dates = pd.to_datetime(df["so_date"], errors="coerce")
    res.stats = {
        "Rows read": report.row_count_raw,
        "Rows usable": report.row_count_clean,
        "Rows in scope": len(in_scope),
        "Customers": int(in_scope["cust_name"].nunique()),
        "Parts": int(in_scope["part_code"].nunique()),
        "Earliest SO": so_dates.min().strftime("%d-%b-%Y") if so_dates.notna().any() else "-",
        "Latest SO": so_dates.max().strftime("%d-%b-%Y") if so_dates.notna().any() else "-",
    }

    live = _live_snapshot()
    if live["max_so_date"] is not None and so_dates.notna().any() and so_dates.max() < live["max_so_date"]:
        res.attention.append(
            f"This file's latest SO date ({so_dates.max():%d-%b-%Y}) is OLDER than the data currently loaded "
            f"({live['max_so_date']:%d-%b-%Y}). Confirm this is the intended, most recent export.")
    if live["row_count"] and len(in_scope) < 0.8 * live["row_count"]:
        res.attention.append(
            f"This file has {len(in_scope):,} in-scope rows versus {live['row_count']:,} currently loaded - a drop of "
            f"more than 20%. The export may be partial; the Sales Register should be the full history each time.")
    return res


def validate_item_master(path: Path) -> ValidationResult:
    res = ValidationResult(kind="Item Master")
    err = _extension_error(path)
    if err:
        res.errors.append(err)
        return res
    try:
        df, report = load_item_master(str(path))
    except ValueError as e:
        res.errors.append(str(e))
        return res
    except Exception as e:
        res.errors.append(f"The file could not be read as an Item Master: {type(e).__name__}: {e}")
        return res
    if len(df) == 0:
        res.errors.append("No usable part rows were found in the Item Master.")
        return res
    res.warnings.extend(report.issues)
    res.stats = {"Parts": len(df), "With stock > 0": int((df.get("current_stock", pd.Series(dtype=float)) > 0).sum())}
    return res


def validate_part_transfer(path: Path) -> ValidationResult:
    """Part transfer list (Etch unique list). Reuses part_transfer.load_part_transfer, so the
    rules are defined once and the pipeline applies exactly what is validated here."""
    res = ValidationResult(kind="Part Transfer List")
    if path.suffix.lower() not in (".xlsx", ".xlsm"):
        res.errors.append(f"Unsupported file type '{path.suffix}'. Please upload the transfer list as an .xlsx file.")
        return res
    if path.stat().st_size == 0:
        res.errors.append("The uploaded file is empty.")
        return res
    try:
        df, report = load_part_transfer(path)
    except Exception as e:
        res.errors.append(f"The file could not be read as a part transfer list: {type(e).__name__}: {e}")
        return res
    res.errors.extend(report["errors"])
    if res.errors:
        return res
    res.warnings.extend(report["warnings"] + report["notes"])
    res.stats = {
        "Parts": len(df),
        "Transfer dates": len(report["stats"].get("transfer_dates", [])),
        "Cannot be priced": int(df["corp_list_price_cannot_be_priced"].sum()),
        "Incomplete rows": int(df["incomplete_row"].sum()),
    }
    live = existing_reference_counts()["part_transfer_parts"]
    if live and len(df) < 0.8 * live:
        res.attention.append(
            f"This file has {len(df):,} parts versus {live:,} currently loaded - a drop of more than 20%. "
            f"Confirm this is the complete transfer list, not a partial one.")
    return res


def validate_commitments(path: Path) -> ValidationResult:
    res = ValidationResult(kind="Customer Commitments")
    err = _extension_error(path)
    if err:
        res.errors.append(err)
        return res
    try:
        df = load_customer_commitments(str(path))
    except Exception as e:
        res.errors.append(str(e))
        return res
    if len(df) == 0:
        res.errors.append("No valid commitment rows found (check Month and Committed Qty values).")
        return res
    res.stats = {"Commitments": len(df), "Customers": int(df["cust_name"].nunique()), "Parts": int(df["part_code"].nunique())}
    return res


# ---------------------------------------------------------------------------
# Refresh history / status
# ---------------------------------------------------------------------------
def _read_log() -> list:
    try:
        return json.loads(REFRESH_LOG_PATH.read_text())
    except Exception:
        return []


def _append_log(entry: dict):
    log = _read_log()
    log.insert(0, entry)
    REFRESH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFRESH_LOG_PATH.write_text(json.dumps(log[:MAX_HISTORY], indent=2))


def get_refresh_history(n: int = 10) -> pd.DataFrame:
    log = _read_log()[:n]
    if not log:
        return pd.DataFrame(columns=["When", "Status", "Source file", "Duration (s)", "Detail"])
    return pd.DataFrame([{
        "When": e.get("timestamp"), "Status": e.get("status"), "Source file": e.get("source_file"),
        "Duration (s)": e.get("duration_seconds"), "Detail": e.get("detail"),
    } for e in log])


def get_last_successful_refresh() -> dict | None:
    for e in _read_log():
        if e.get("status") == "Success":
            return e
    return None


def existing_reference_counts() -> dict:
    """How much Item Master / commitment data is already in the live database (reused if not re-uploaded)."""
    out = {"item_master_parts": 0, "commitments": 0, "part_transfer_parts": 0}
    if not DB_PATH.exists():
        return out
    try:
        conn = sqlite3.connect(DB_PATH)
        for key, table in (("item_master_parts", "item_master"), ("commitments", "customer_commitments"),
                            ("part_transfer_parts", "part_transfer")):
            try:
                out[key] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except Exception:
                pass
        conn.close()
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Locking - one refresh at a time
# ---------------------------------------------------------------------------
def _acquire_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists() and time.time() - LOCK_PATH.stat().st_mtime > STALE_LOCK_SECONDS:
        LOCK_PATH.unlink(missing_ok=True)  # leftover from a crashed run
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        raise RefreshBusy("Another data refresh is already in progress. Please wait for it to finish.")


def _release_lock():
    LOCK_PATH.unlink(missing_ok=True)


def is_refresh_running() -> bool:
    return LOCK_PATH.exists() and time.time() - LOCK_PATH.stat().st_mtime <= STALE_LOCK_SECONDS


# ---------------------------------------------------------------------------
# The refresh itself
# ---------------------------------------------------------------------------
class _LineWriter:
    """File-like object forwarding each complete printed line to a callback (for live status in the UI)."""
    def __init__(self, callback):
        self.callback, self.buf = callback, ""

    def write(self, text):
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.callback(line)
        return len(text)

    def flush(self):
        pass


def _load_live_table(table: str):
    if not DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        conn.close()
        return df if len(df) else None
    except Exception:
        return None


def _verify_staging(staging_db: Path):
    conn = sqlite3.connect(staging_db)
    try:
        for table in CRITICAL_TABLES:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if n == 0:
                raise RuntimeError(f"Verification failed: table '{table}' is empty after processing.")
        # once a part transfer list exists, the combined part master must have been built from it
        has_transfer = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='part_transfer'").fetchone()[0]
        if has_transfer and conn.execute("SELECT COUNT(*) FROM part_transfer").fetchone()[0] > 0:
            if conn.execute("SELECT COUNT(*) FROM part_master").fetchone()[0] == 0:
                raise RuntimeError("Verification failed: table 'part_master' is empty after processing.")
    except sqlite3.OperationalError as e:
        raise RuntimeError(f"Verification failed: a required table is missing ({e}).")
    finally:
        conn.close()


def _swap_into_place(staging_db: Path):
    """Atomically replace the live database. Retries briefly in case a reader has it open (Windows)."""
    last_err = None
    for _ in range(8):
        try:
            os.replace(staging_db, DB_PATH)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"Could not replace the live database (it may be locked by another program): {last_err}")


def refresh_data(sales_register_path: Path, source_name: str, item_master_path: Path = None,
                  commitments_path: Path = None, transfer_path: Path = None, log_fn=print) -> dict:
    """
    Runs the full existing pipeline against a staging database and, only if
    everything succeeds and verifies, swaps it into place as the live data.
    Raises on failure - and in that case the live database is untouched.
    """
    _acquire_lock()
    started = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # staging lives beside the live DB so the final swap is always a same-drive atomic replace
    staging_dir = Path(tempfile.mkdtemp(prefix="staging_", dir=DATA_DIR))
    staging_db = staging_dir / "sciip_staging.db"
    try:
        if DB_PATH.exists():
            shutil.copy2(DB_PATH, staging_db)  # start from live so the append-only forecast log carries over

        reuse_item_master = None if item_master_path else _load_live_table("item_master")
        reuse_commitments = None if commitments_path else _load_live_table("customer_commitments")
        reuse_transfer = (not transfer_path) and existing_reference_counts()["part_transfer_parts"] > 0

        with contextlib.redirect_stdout(_LineWriter(log_fn)):
            _, report_path = run_pipeline(
                str(sales_register_path),
                item_master_filepath=str(item_master_path) if item_master_path else None,
                customer_commitments_filepath=str(commitments_path) if commitments_path else None,
                db_path=staging_db, output_dir=OUTPUT_DIR,
                item_master_df=reuse_item_master, customer_commitments_df=reuse_commitments,
                part_transfer_filepath=str(transfer_path) if transfer_path else None,
            )

        log_fn("Verifying processed data...")
        _verify_staging(staging_db)
        log_fn("Publishing refreshed data...")
        _swap_into_place(staging_db)

        conn = sqlite3.connect(DB_PATH)
        n_lines = conn.execute("SELECT COUNT(*) FROM demand_lines").fetchone()[0]
        n_parts = conn.execute("SELECT COUNT(*) FROM part_intelligence_summary").fetchone()[0]
        n_customers = conn.execute("SELECT COUNT(*) FROM customer_profiles").fetchone()[0]
        conn.close()

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": "Success",
            "source_file": source_name, "duration_seconds": round(time.time() - started, 1),
            "detail": f"{n_lines:,} demand lines, {n_parts:,} parts, {n_customers:,} customers"
                      + (" | Item Master updated" if item_master_path
                         else " | Item Master reused" if reuse_item_master is not None
                         else " | No Item Master loaded")
                      + (" | Commitments updated" if commitments_path else "")
                      + (" | Part transfer list updated" if transfer_path
                         else " | Part transfer list reused" if reuse_transfer else ""),
            "report": str(report_path),
        }
        _append_log(entry)
        return entry
    except Exception as e:
        _append_log({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": "Failed",
            "source_file": source_name, "duration_seconds": round(time.time() - started, 1),
            "detail": f"{type(e).__name__}: {e} (live data unchanged)",
        })
        e.sciip_traceback = traceback.format_exc()
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        _release_lock()
