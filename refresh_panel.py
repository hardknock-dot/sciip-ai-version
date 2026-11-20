"""
SCIIP - Data Refresh panel (presentation only)
Reusable Streamlit component: upload -> validate -> review -> refresh.
All processing is delegated to refresh.py, which in turn runs the existing
pipeline. Nothing here calculates or transforms business data.
"""
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st

import refresh

STALE_AFTER_DAYS = 7
PIPELINE_STEP_COUNT = 8  # Step 1, 2, 3, 3b, 3c, 4, 4b, 5 - used only for the progress bar

UPLOADERS = {
    "sales": {
        "label": "Sales Register", "required": True, "validator": refresh.validate_sales_register,
        "help": "Latest SAP Sales Register export - the full history every time (e.g. 1-Jan-2025 to today).",
    },
    "item": {
        "label": "Item Master", "required": False, "validator": refresh.validate_item_master,
        "help": "Optional. Upload to update stock, lead times, vendor orders and purchasing constraints. "
                "If skipped, the Item Master already loaded is reused.",
    },
    "commit": {
        "label": "Customer Commitments", "required": False, "validator": refresh.validate_commitments,
        "help": "Optional. Customer-provided forecast quantities (Project, Part Code, Customer, Month, Committed Qty). "
                "If skipped, commitments already loaded are reused.",
    },
    "transfer": {
        "label": "Part Transfer List", "required": False, "validator": refresh.validate_part_transfer,
        "types": ["xlsx"],
        "help": "Optional. The Etch unique list (transfer-related part details). Merged side by side with the Item Master "
                "into the Part Master. If skipped, the transfer list already loaded is reused.",
    },
}


# ---------------------------------------------------------------------------
# Freshness indicator (used by the dashboard's top bar)
# ---------------------------------------------------------------------------
def freshness_info(fallback_timestamp: str = None) -> dict:
    """Latest successful refresh time, source file, age and stale flag."""
    last = refresh.get_last_successful_refresh()
    ts_text, source = None, None
    if last:
        ts_text, source = last.get("timestamp"), last.get("source_file")
    elif fallback_timestamp:
        ts_text = fallback_timestamp  # data loaded via the command line before this feature existed

    if not ts_text:
        return {"label": "NO DATA LOADED YET", "stale": True, "source": None}
    try:
        ts = datetime.fromisoformat(str(ts_text).replace(" ", "T"))
    except ValueError:
        return {"label": f"DATA AS OF {ts_text}", "stale": False, "source": source}

    age_days = (datetime.now() - ts).days
    label = f"DATA REFRESHED {ts:%d-%b-%Y %H:%M}"
    if source:
        label += f" \u00b7 {source}"
    if age_days >= STALE_AFTER_DAYS:
        label += f" \u00b7 {age_days} DAYS OLD"
    return {"label": label, "stale": age_days >= STALE_AFTER_DAYS, "source": source}


def show_pending_notifications():
    """Toast shown once after a successful refresh, whichever tab is visible."""
    notice = st.session_state.pop("refresh_notice", None)
    if notice:
        st.toast(f"Data refreshed successfully - {notice['detail']}", icon="\u2705")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _gen() -> int:
    return st.session_state.setdefault("refresh_gen", 0)


def _validated_entry(kind: str, uploaded):
    """Validate an upload once per distinct file (cached across Streamlit reruns)."""
    cache = st.session_state.setdefault("refresh_validation", {})
    if uploaded is None:
        cache.pop(kind, None)
        return None
    data = uploaded.getvalue()
    sig = hashlib.md5(data).hexdigest()
    entry = cache.get(kind)
    if entry and entry["sig"] == sig:
        return entry
    cfg = UPLOADERS[kind]
    path = refresh.save_upload(data, uploaded.name, kind)
    with st.spinner(f"Validating {cfg['label']}..."):
        result = cfg["validator"](path)
    entry = {"sig": sig, "path": path, "name": uploaded.name, "result": result}
    cache[kind] = entry
    return entry


def _stat_cards(stats: dict):
    if not stats:
        return
    cols = st.columns(len(stats))
    for col, (k, v) in zip(cols, stats.items()):
        value = f"{v:,}" if isinstance(v, (int, float)) and not isinstance(v, bool) else v
        col.markdown(f'<div class="scp-kpi" style="--kpi-accent:#06B6D4;"><div class="scp-kpi-label">{k}</div>'
                     f'<div class="scp-kpi-value" style="font-size:22px;">{value}</div></div>', unsafe_allow_html=True)


def _render_validation(entry):
    res = entry["result"]
    if res.ok:
        st.markdown(f'<span class="scp-pill scp-pill-active">VALIDATED</span> '
                    f'<span class="scp-mono" style="font-size:12px;color:#5B6B82;">{entry["name"]}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="scp-pill scp-pill-lumpy">REJECTED</span> '
                    f'<span class="scp-mono" style="font-size:12px;color:#5B6B82;">{entry["name"]}</span>', unsafe_allow_html=True)
    for e in res.errors:
        st.error(e)
    if res.ok:
        _stat_cards(res.stats)
    for a in res.attention:
        st.warning(a)
    if res.warnings:
        with st.expander(f"{len(res.warnings)} validation note(s) - non-blocking"):
            for w in res.warnings:
                st.caption(f"\u2022 {w}")


def _current_status_card():
    last = refresh.get_last_successful_refresh()
    refs = refresh.existing_reference_counts()
    info = freshness_info()
    color = "#F59E0B" if info["stale"] else "#10B981"
    st.markdown('<div class="scp-section-label">Current data</div>', unsafe_allow_html=True)
    if last:
        st.markdown(
            f'<div class="scp-kpi"><div class="scp-kpi-label">Last successful refresh</div>'
            f'<div class="scp-kpi-value" style="font-size:20px;color:{color};">{last["timestamp"]}</div>'
            f'<div class="scp-kpi-sub">Source: {last.get("source_file")} \u00b7 {last.get("detail")}</div></div>',
            unsafe_allow_html=True)
    else:
        st.info("No refresh has been run from the dashboard yet. Data loaded via the command line (if any) is still shown.")
    st.caption(f"Reference data on file: Item Master {refs['item_master_parts']:,} parts \u00b7 "
               f"Part transfer list {refs['part_transfer_parts']:,} parts \u00b7 "
               f"Customer commitments {refs['commitments']:,} rows (reused automatically unless you upload new ones)")


def _run_refresh(entries: dict):
    status = st.status("Starting data refresh...", expanded=True)
    progress = status.progress(0)
    log_box = status.empty()
    lines, steps = [], [0]

    def on_line(line: str):
        lines.append(line)
        if line.startswith("--- Step"):
            steps[0] += 1
            status.update(label=f"Processing \u2014 {line.strip('- ').strip()}")
            progress.progress(min(steps[0] / PIPELINE_STEP_COUNT, 0.9))
        elif line.startswith("Verifying"):
            status.update(label="Verifying processed data...")
            progress.progress(0.93)
        elif line.startswith("Publishing"):
            status.update(label="Publishing refreshed data...")
            progress.progress(0.97)
        log_box.code("\n".join(lines[-12:]), language=None)

    sales = entries["sales"]
    try:
        result = refresh.refresh_data(
            sales["path"], sales["name"],
            item_master_path=entries["item"]["path"] if entries.get("item") else None,
            commitments_path=entries["commit"]["path"] if entries.get("commit") else None,
            transfer_path=entries["transfer"]["path"] if entries.get("transfer") else None,
            log_fn=on_line)
    except refresh.RefreshBusy as e:
        status.update(label="Another refresh is already running", state="error", expanded=True)
        st.warning(str(e))
        return
    except Exception as e:
        status.update(label="Refresh failed \u2014 previous data has been kept", state="error", expanded=True)
        st.error(f"The refresh could not be completed: {e}")
        st.caption("Nothing was changed - the dashboard is still showing the last successful dataset.")
        with st.expander("Technical details"):
            st.code(getattr(e, "sciip_traceback", str(e)), language=None)
        return

    progress.progress(1.0)
    status.update(label="Refresh complete", state="complete", expanded=False)
    st.session_state["refresh_notice"] = {"detail": result["detail"]}
    st.session_state["refresh_gen"] = _gen() + 1      # resets the upload widgets
    st.session_state.pop("refresh_validation", None)
    st.cache_data.clear()                              # every dependent section re-reads the new data
    st.rerun()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def render_refresh_panel(key_prefix: str = "refresh"):
    """Renders the full upload -> validate -> refresh experience."""
    st.markdown('<div class="scp-panel">', unsafe_allow_html=True)
    _current_status_card()
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="scp-section-label">Upload latest data</div>', unsafe_allow_html=True)
    st.markdown('<div class="scp-hint">Files are validated first. Your current data is only replaced if processing '
                'completes successfully - otherwise nothing changes.</div>', unsafe_allow_html=True)

    entries = {}
    for kind, cfg in UPLOADERS.items():
        tag = "REQUIRED" if cfg["required"] else "OPTIONAL"
        st.markdown(f'<div class="scp-section-label" style="margin-top:14px;">{cfg["label"]} \u00b7 {tag}</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(cfg["label"], type=cfg.get("types", ["xlsx", "xls", "csv"]), key=f"{key_prefix}_{kind}_{_gen()}",
                                      label_visibility="collapsed", help=cfg["help"])
        if uploaded is None:
            st.caption(cfg["help"])
        entry = _validated_entry(kind, uploaded)
        if entry:
            _render_validation(entry)
            entries[kind] = entry

    # ---- readiness ----------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    sales = entries.get("sales")
    blockers, attention = [], []
    if refresh.is_refresh_running():
        blockers.append("Another refresh is currently running.")
    if not sales:
        blockers.append("Upload a Sales Register to begin.")
    for kind, entry in entries.items():
        if not entry["result"].ok:
            blockers.append(f"{UPLOADERS[kind]['label']} failed validation - fix or remove it.")
        attention.extend(entry["result"].attention)

    acknowledged = True
    if attention and not blockers:
        acknowledged = st.checkbox("I have reviewed the warnings above and want to proceed",
                                     key=f"{key_prefix}_ack_{_gen()}")

    can_run = not blockers and acknowledged
    if blockers:
        for b in blockers:
            st.caption(f"\u26a0 {b}")
    if st.button("Run data refresh", type="primary", disabled=not can_run, key=f"{key_prefix}_run_{_gen()}"):
        _run_refresh(entries)

    # ---- history ------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="scp-section-label">Refresh history</div>', unsafe_allow_html=True)
    history = refresh.get_refresh_history(10)
    if len(history):
        st.dataframe(history, use_container_width=True, hide_index=True, height=250)
    else:
        st.caption("No refreshes recorded yet.")
    st.markdown('</div>', unsafe_allow_html=True)
