"""
SCIIP Dashboard - Enterprise Supply Chain & Inventory Intelligence Command Center
Run: streamlit run dashboard.py
Reads directly from data/sciip.db. Requires Streamlit 1.35+ for row selection.
"""
import sqlite3
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import refresh_panel  # upload + validate + refresh UI (delegates to the existing pipeline)
import part_transfer_integration  # Item Master + part transfer list side by side (part_master table)
import copilot  # grounded AI Copilot; reads existing SQLite tables only

DB_PATH = Path(__file__).parent / "data" / "sciip.db"
st.set_page_config(
    page_title="SCIIP | Supply Chain Intelligence Hub",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# Modern, high-visibility dashboard color palette
PATTERN_COLOR = {"Smooth": "#10B981", "Intermittent": "#F59E0B", "Erratic": "#06B6D4", "Lumpy": "#EF4444"}
STATUS_COLOR = {"CLOSED": "#10B981", "OPEN": "#06B6D4"}
COMPARE_COLORS = ["#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#3B82F6"]

# ---------------------------------------------------------------------------
# Global Design System CSS: Deep Slate Command Center Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg-primary: #0A0E17;
  --bg-surface: #111827;
  --bg-elevated: #1E293B;
  --border-subtle: rgba(255, 255, 255, 0.08);
  --border-card: #1F2937;
  --border-focus: #06B6D4;
  --text-primary: #F8FAFC;
  --text-secondary: #94A3B8;
  --text-muted: #64748B;
  --accent-cyan: #06B6D4;
  --accent-teal: #0D9488;
  --accent-emerald: #10B981;
  --accent-amber: #F59E0B;
  --accent-rose: #EF4444;
  --accent-indigo: #6366F1;
  --accent-blue: #3B82F6;
  --panel: #111827;
  --elevated: #1E293B;
  --border: rgba(255, 255, 255, 0.08);
  --text: #F8FAFC;
  --muted: #94A3B8;
  --sage: #06B6D4;
  --green: #10B981;
  --amber: #F59E0B;
  --red: #EF4444;
}

* { box-sizing: border-box; }

.stApp {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: 'Inter', -apple-system, sans-serif;
}

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stAppViewContainer"] > .main {
  padding-top: 14px;
}

[data-testid="stMainBlockContainer"],
[data-testid="stAppViewContainer"] .main .block-container {
  max-width: 1720px !important;
  width: 95% !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
  padding-top: 8px !important;
  margin: 0 auto !important;
}

/* Typography */
h1, h2, h3, h4, h5, h6, .scp-title, .scp-wordmark {
  font-family: 'Space Grotesk', -apple-system, sans-serif !important;
  color: var(--text-primary) !important;
  letter-spacing: -0.01em;
}

body, p, div, span, label, small {
  font-family: 'Inter', -apple-system, sans-serif;
  color: var(--text-primary);
}

.stMarkdown, .stMarkdown p, .stMarkdown span,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
  color: var(--text-secondary);
}

.scp-mono {
  font-family: 'JetBrains Mono', 'IBM Plex Mono', monospace !important;
}

/* Top Navigation Bar */
.scp-topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  background: linear-gradient(180deg, rgba(22, 30, 48, 0.85) 0%, rgba(17, 24, 39, 0.92) 100%);
  backdrop-filter: blur(14px);
  margin-bottom: 20px;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45);
}

.scp-brand-block {
  display: flex;
  align-items: center;
  gap: 16px;
  min-width: 0;
}

.scp-brand-icon {
  width: 44px;
  height: 44px;
  border-radius: 10px;
  background: linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(99, 102, 241, 0.25) 100%);
  border: 1px solid rgba(6, 182, 212, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #38BDF8;
  box-shadow: 0 0 20px rgba(6, 182, 212, 0.25);
  flex-shrink: 0;
}

.scp-wordmark {
  font-size: 26px;
  font-weight: 700;
  letter-spacing: 0.05em;
  color: var(--text-primary);
  display: flex;
  align-items: center;
  gap: 2px;
}

.scp-wordmark span {
  background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.scp-badge-tag {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  padding: 3px 8px;
  border-radius: 4px;
  background: rgba(6, 182, 212, 0.15);
  color: #38BDF8;
  border: 1px solid rgba(6, 182, 212, 0.35);
  text-transform: uppercase;
  margin-left: 10px;
}

.scp-subtitle {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 3px;
  letter-spacing: 0.02em;
}

.scp-meta {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 6px;
  max-width: 45%;
}

.scp-freshness-pill {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 12px;
  border-radius: 20px;
  font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 500;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(15, 23, 42, 0.7);
}

.scp-pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  position: relative;
  flex-shrink: 0;
}

.scp-pulse-dot::after {
  content: '';
  position: absolute;
  top: -3px;
  left: -3px;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: inherit;
  opacity: 0.45;
  animation: scpPulse 2s infinite ease-out;
}

@keyframes scpPulse {
  0% { transform: scale(0.8); opacity: 0.85; }
  100% { transform: scale(2.3); opacity: 0; }
}

.scp-source {
  font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 480px;
}

/* Executive Snapshot Cards */
.scp-snapshot-header {
  margin: 18px 0 12px 0;
}

.scp-overview-title {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.01em;
  margin: 0 0 3px 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.scp-overview-subtitle {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 16px;
}

.scp-kpi {
  position: relative;
  overflow: hidden;
  background: linear-gradient(180deg, rgba(26, 36, 56, 0.75) 0%, rgba(17, 24, 39, 0.88) 100%);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  padding: 18px 20px;
  min-height: 138px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.2s ease;
  backdrop-filter: blur(10px);
}

.scp-kpi::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--kpi-accent, var(--accent-cyan));
  opacity: 0.9;
}

.scp-kpi:hover {
  transform: translateY(-2px);
  border-color: rgba(6, 182, 212, 0.4);
  box-shadow: 0 10px 28px -5px rgba(0, 0, 0, 0.55), 0 0 16px rgba(6, 182, 212, 0.15);
}

.scp-kpi-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.scp-kpi-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-secondary);
  font-weight: 600;
}

.scp-kpi-icon {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.scp-kpi-value {
  font-family: 'Space Grotesk', -apple-system, sans-serif;
  font-size: 30px;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1.1;
  letter-spacing: -0.02em;
}

.scp-kpi-sub {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 5px;
  font-weight: 500;
}

/* Panels */
.scp-panel {
  background: rgba(17, 24, 39, 0.7);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  padding: 24px;
  margin-top: 14px;
  backdrop-filter: blur(10px);
}

.scp-section-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--text-secondary);
  margin-bottom: 12px;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 6px;
}

.scp-section-label strong {
  color: var(--text-primary);
}

.scp-desc {
  color: var(--text-secondary);
  font-size: 13px;
  margin: 0 0 14px;
  line-height: 1.5;
}

.scp-hint {
  color: var(--text-muted);
  font-size: 12px;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.scp-subsection {
  border-top: 1px solid var(--border-subtle);
  padding-top: 18px;
  margin-top: 20px;
}

.scp-row-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 14px;
  border-bottom: 1px solid var(--border-subtle);
  font-size: 13px;
  border-radius: 6px;
  transition: background 0.15s ease;
}

.scp-row-item:hover {
  background: rgba(255, 255, 255, 0.03);
}

/* Scope Selector (Radio Buttons) */
div[data-testid="stRadio"] {
  background: rgba(17, 24, 39, 0.75);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 6px 8px;
  backdrop-filter: blur(8px);
}

div[data-testid="stRadio"] div[role="radiogroup"] {
  gap: 6px;
}

div[data-testid="stRadio"] div[role="radiogroup"] label {
  padding: 7px 16px;
  border: 1px solid transparent;
  border-radius: 7px;
  font-size: 12px;
  font-weight: 500;
  color: var(--text-secondary);
  transition: all 0.15s ease;
}

div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
  background: rgba(255, 255, 255, 0.04);
  color: var(--text-primary);
}

div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(135deg, #0891B2 0%, #0284C7 100%);
  border-color: rgba(6, 182, 212, 0.5);
  box-shadow: 0 2px 10px rgba(8, 145, 178, 0.35);
}

div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) span {
  color: #FFFFFF !important;
  font-weight: 600;
}

/* Modern Segmented Navigation Tabs */
div[data-baseweb="tab-list"] {
  gap: 6px;
  border-bottom: 1px solid var(--border-subtle);
  padding: 6px 8px;
  background: rgba(15, 23, 42, 0.7);
  border-radius: 12px 12px 0 0;
  overflow-x: auto;
}

div[data-baseweb="tab"] {
  padding: 10px 18px;
  color: var(--text-secondary);
  border-radius: 8px;
  font-size: 13px !important;
  font-weight: 500;
  border: 1px solid transparent;
  transition: all 0.18s ease;
}

div[data-baseweb="tab"]:hover {
  color: var(--text-primary);
  background: rgba(255, 255, 255, 0.04);
}

div[data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, rgba(6, 182, 212, 0.18) 0%, rgba(59, 130, 246, 0.12) 100%) !important;
  border: 1px solid rgba(6, 182, 212, 0.4) !important;
  box-shadow: 0 2px 12px rgba(6, 182, 212, 0.15) !important;
}

div[data-baseweb="tab"][aria-selected="true"],
div[data-baseweb="tab"][aria-selected="true"] p,
div[data-baseweb="tab"][aria-selected="true"] span {
  color: #38BDF8 !important;
  font-weight: 600;
}

div[data-baseweb="tab-highlight"] {
  display: none !important;
}

/* Micro-Badges & Status Pills */
.scp-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.02em;
  white-space: nowrap;
}

.scp-pill-smooth, .scp-pill-active, .scp-pill-growing, .scp-pill-closed, .scp-pill-adequate {
  background: rgba(16, 185, 129, 0.15);
  color: #34D399;
  border: 1px solid rgba(16, 185, 129, 0.35);
}

.scp-pill-intermittent, .scp-pill-duesoon, .scp-pill-replenishsoon, .scp-pill-medium {
  background: rgba(245, 158, 11, 0.15);
  color: #FBBF24;
  border: 1px solid rgba(245, 158, 11, 0.35);
}

.scp-pill-erratic, .scp-pill-stable, .scp-pill-open {
  background: rgba(6, 182, 212, 0.15);
  color: #38BDF8;
  border: 1px solid rgba(6, 182, 212, 0.35);
}

.scp-pill-lumpy, .scp-pill-overdue, .scp-pill-declining, .scp-pill-replenishnow, .scp-pill-high {
  background: rgba(239, 68, 68, 0.15);
  color: #F87171;
  border: 1px solid rgba(239, 68, 68, 0.35);
}

.scp-pill-new, .scp-pill-unknown {
  background: rgba(148, 163, 184, 0.12);
  color: #CBD5E1;
  border: 1px solid rgba(148, 163, 184, 0.25);
}

/* Executive Dossier Header & Cards */
.scp-dossier-card {
  background: linear-gradient(180deg, rgba(26, 36, 56, 0.75) 0%, rgba(17, 24, 39, 0.88) 100%);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  padding: 22px;
  margin-bottom: 20px;
}

.scp-dossier-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 18px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-subtle);
}

.scp-avatar {
  width: 46px;
  height: 46px;
  border-radius: 12px;
  background: linear-gradient(135deg, #0284C7 0%, #6366F1 100%);
  color: #FFFFFF;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 17px;
  box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
  flex-shrink: 0;
}

.scp-avatar-part {
  background: linear-gradient(135deg, #0D9488 0%, #06B6D4 100%);
  box-shadow: 0 4px 14px rgba(6, 182, 212, 0.35);
}

.scp-dossier-title {
  font-size: 20px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.scp-dossier-badges {
  display: flex;
  align-items: center;
  gap: 8px;
}

.scp-meta-chip {
  font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--text-muted);
  background: rgba(255, 255, 255, 0.04);
  padding: 3px 8px;
  border-radius: 4px;
  border: 1px solid var(--border-subtle);
}

/* Empty States */
.scp-empty-state {
  text-align: center;
  padding: 55px 24px;
  border: 1px dashed var(--border-subtle);
  border-radius: 12px;
  background: rgba(17, 24, 39, 0.35);
}

.scp-empty-icon {
  font-size: 36px;
  margin-bottom: 12px;
  opacity: 0.8;
}

.scp-empty-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 6px;
}

.scp-empty-desc {
  font-size: 13px;
  color: var(--text-muted);
  max-width: 440px;
  margin: 0 auto;
  line-height: 1.5;
}

/* Form controls & Buttons */
button[kind="primary"] {
  background: linear-gradient(135deg, #0891B2 0%, #0284C7 100%) !important;
  border: 1px solid rgba(6, 182, 212, 0.45) !important;
  color: #FFFFFF !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 14px rgba(8, 145, 178, 0.3) !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

button[kind="primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 18px rgba(8, 145, 178, 0.45) !important;
}

button[kind="secondary"] {
  background: rgba(30, 41, 59, 0.7) !important;
  border: 1px solid var(--border-subtle) !important;
  color: var(--text-primary) !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  transition: all 0.15s ease !important;
}

button[kind="secondary"]:hover {
  border-color: rgba(6, 182, 212, 0.4) !important;
  color: #38BDF8 !important;
  background: rgba(30, 41, 59, 0.9) !important;
}

input, textarea, div[data-baseweb="select"] > div {
  border-radius: 8px !important;
  border-color: var(--border-subtle) !important;
  background: rgba(17, 24, 39, 0.8) !important;
  color: var(--text-primary) !important;
}

div[data-testid="stPlotlyChart"] {
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 12px;
  background: rgba(17, 24, 39, 0.55);
  backdrop-filter: blur(8px);
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  overflow: hidden;
  background: rgba(17, 24, 39, 0.55);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
  background: #0E131F !important;
  border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] .scp-sidebar-header {
  padding: 14px 0 18px 0;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 18px;
}

section[data-testid="stSidebar"] .scp-sidebar-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

section[data-testid="stSidebar"] .scp-sidebar-card {
  background: rgba(17, 24, 39, 0.6);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 14px;
  margin-top: 14px;
}

/* Floating Copilot Launcher & Drawer */
div[data-testid="stMarkdownContainer"]:has(.scp-launcher-anchor) { display: none; }
div[data-testid="stMarkdownContainer"]:has(.scp-launcher-anchor) ~ div[data-testid="stElementContainer"]:has(button) {
  position: fixed !important;
  right: 28px;
  bottom: 28px;
  z-index: 1000;
  width: auto !important;
}
div[data-testid="stMarkdownContainer"]:has(.scp-launcher-anchor) ~ div[data-testid="stElementContainer"]:has(button) button {
  min-width: 54px !important;
  height: 54px !important;
  padding: 0 18px !important;
  border-radius: 27px !important;
  background: linear-gradient(135deg, #0891B2 0%, #6366F1 100%) !important;
  border: 1px solid rgba(255, 255, 255, 0.25) !important;
  box-shadow: 0 8px 30px rgba(8, 145, 178, 0.4) !important;
  color: #FFFFFF !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700 !important;
  font-size: 16px !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease !important;
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
}
div[data-testid="stMarkdownContainer"]:has(.scp-launcher-anchor) ~ div[data-testid="stElementContainer"]:has(button) button:hover {
  transform: translateY(-2px) scale(1.03) !important;
  box-shadow: 0 12px 35px rgba(8, 145, 178, 0.55) !important;
}

div[data-testid="stDialog"] { z-index: 1100 !important; }
div[role="dialog"] {
  position: fixed !important;
  top: 24px !important;
  right: 24px !important;
  bottom: 24px !important;
  left: auto !important;
  width: 460px !important;
  max-width: calc(100vw - 48px) !important;
  height: calc(100vh - 48px) !important;
  max-height: none !important;
  overflow-y: auto !important;
  padding: 24px !important;
  background: #111827 !important;
  border: 1px solid rgba(6, 182, 212, 0.35) !important;
  border-radius: 12px !important;
  box-shadow: 0 25px 65px rgba(0, 0, 0, 0.7) !important;
  animation: scp-drawer-in .2s ease-out;
}
@keyframes scp-drawer-in {
  from { opacity: 0; transform: translateX(20px); }
  to { opacity: 1; transform: translateX(0); }
}

.scp-drawer-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  padding-bottom: 16px;
  margin-bottom: 14px;
  border-bottom: 1px solid var(--border-subtle);
}
.scp-drawer-title { font: 600 20px 'Space Grotesk', sans-serif; color: var(--text-primary); }
.scp-drawer-subtitle { margin-top: 4px; font-size: 12px; color: var(--text-secondary); }
.scp-drawer-status {
  padding: 3px 8px;
  border: 1px solid rgba(16, 185, 129, 0.4);
  border-radius: 4px;
  font: 10px 'JetBrains Mono', monospace;
  color: #34D399;
  background: rgba(16, 185, 129, 0.12);
  letter-spacing: .08em;
}
.scp-drawer-empty { padding: 30px 12px 20px; text-align: center; }
.scp-drawer-empty-title { font: 600 16px 'Space Grotesk', sans-serif; color: var(--text-primary); }
.scp-drawer-empty-copy { margin-top: 8px; font-size: 12px; color: var(--text-secondary); line-height: 1.5; }
div[role="dialog"] [data-testid="stChatMessage"] {
  background: rgba(30, 41, 59, 0.65);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  padding: 10px 14px;
  margin: 10px 0;
}
div[role="dialog"] [data-testid="stChatMessage"] p { font-size: 13px; line-height: 1.5; }
div[role="dialog"] [data-testid="stButton"] button { min-height: 38px; font-size: 12px !important; }

@media (max-width: 1000px) {
  [data-testid="stMainBlockContainer"], [data-testid="stAppViewContainer"] .main .block-container { width: 96% !important; }
  .scp-brand-block { align-items: flex-start; flex-direction: column; gap: 4px; }
  .scp-meta { max-width: 50%; }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Database Loader & Utility Functions
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_table(name):
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(f"SELECT * FROM {name}", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

def get_freshness():
    df = load_table("demand_lines")
    if len(df) == 0 or "_computed_at" not in df.columns:
        return "no data yet"
    return df["_computed_at"].max()

def pill(label, css_class):
    return f'<span class="scp-pill scp-pill-{css_class}">{label}</span>'

def pattern_pill(p):
    return pill(p, p.lower() if isinstance(p, str) else "unknown")

def status_pill(s):
    key = {"Active": "active", "Due soon": "duesoon", "Overdue / went quiet": "overdue",
           "CLOSED": "closed", "OPEN": "open"}.get(s, "new")
    return pill(s, key)

def replen_pill(s):
    key = {"Replenish Now": "replenishnow", "Replenish Soon": "replenishsoon", "Adequate": "adequate",
           "Obsolete by OEM - Alternate Source Needed": "medium"}.get(s, "unknown")
    return pill(s, key)

def risk_pill(sev):
    key = {"High": "high", "Medium": "medium", "Low": "active", "Opportunity": "growing"}.get(sev, "unknown")
    return pill(sev, key)

def compute_trend_direction(yearly_df):
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
    res = df.apply(calc, axis=1)
    df["trend_direction"] = res.apply(lambda x: x[0])
    df["yoy_change_pct"] = res.apply(lambda x: x[1])
    return df

# Helper to render clean executive KPI cards
def render_kpi(label: str, value: str, subtext: str = "", accent_color: str = "#06B6D4", icon_svg: str = "") -> str:
    sub_html = f'<div class="scp-kpi-sub">{subtext}</div>' if subtext else ""
    icon_html = f'<div class="scp-kpi-icon" style="color:{accent_color}; background:rgba(255,255,255,0.06);">{icon_svg}</div>' if icon_svg else ""
    return f"""
    <div class="scp-kpi" style="--kpi-accent: {accent_color};">
      <div class="scp-kpi-header">
        <span class="scp-kpi-label">{label}</span>
        {icon_html}
      </div>
      <div class="scp-kpi-value">{value}</div>
      {sub_html}
    </div>
    """

# ---------------------------------------------------------------------------
# High-End Plotly Chart Themes
# ---------------------------------------------------------------------------
def waveform_chart(monthly_df, title=""):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_df["month"].astype(str),
        y=monthly_df["qty"],
        marker=dict(
            color="#06B6D4",
            line=dict(color="#0891B2", width=1)
        ),
        hovertemplate="<b>%{x}</b><br>Quantity: <b>%{y:,.0f}</b> units<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk, sans-serif", size=13, color="#94A3B8")),
        plot_bgcolor="rgba(17, 24, 39, 0.4)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F8FAFC", size=11),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(size=10, color="#94A3B8")),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", title="Qty", tickfont=dict(size=10, color="#94A3B8")),
        margin=dict(l=45, r=20, t=32, b=32),
        height=260,
        showlegend=False,
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="rgba(255,255,255,0.15)", font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12))
    )
    return fig

def multi_waveform_chart(series_list):
    fig = go.Figure()
    for i, (label, mdf) in enumerate(series_list):
        c = COMPARE_COLORS[i % len(COMPARE_COLORS)]
        fig.add_trace(go.Bar(
            x=mdf["month"].astype(str),
            y=mdf["qty"],
            name=label,
            marker=dict(color=c),
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Quantity: <b>%{{y:,.0f}}</b><extra></extra>"
        ))
    fig.update_layout(
        plot_bgcolor="rgba(17, 24, 39, 0.4)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F8FAFC", size=11),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(size=10, color="#94A3B8")),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", title="Qty", tickfont=dict(size=10, color="#94A3B8")),
        margin=dict(l=45, r=20, t=25, b=35),
        height=320,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11, color="#CBD5E1")),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="rgba(255,255,255,0.15)", font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12))
    )
    return fig

def yearly_bar_chart(row, year_cols, title=""):
    fig = go.Figure(go.Bar(
        x=year_cols,
        y=[row[y] for y in year_cols],
        marker=dict(color="#3B82F6", line=dict(color="#2563EB", width=1)),
        hovertemplate="<b>%{x}</b><br>Revenue: <b>$%{y:,.0f}</b><extra></extra>"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk, sans-serif", size=12, color="#94A3B8")),
        plot_bgcolor="rgba(17, 24, 39, 0.4)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F8FAFC", size=11),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(size=10, color="#94A3B8")),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(size=10, color="#94A3B8")),
        margin=dict(l=45, r=20, t=32, b=30),
        height=210,
        showlegend=False,
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="rgba(255,255,255,0.15)", font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12))
    )
    return fig

def pattern_distribution_chart(pi_df):
    counts = pi_df["demand_pattern"].value_counts()
    patterns = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
    values = [counts.get(p, 0) for p in patterns]
    fig = go.Figure(go.Bar(
        x=patterns,
        y=values,
        marker=dict(color=[PATTERN_COLOR[p] for p in patterns]),
        text=values,
        textposition="outside",
        textfont=dict(color="#F8FAFC", family="Space Grotesk, sans-serif", size=12),
        hovertemplate="<b>%{x} Pattern</b><br>Part Count: <b>%{y:,}</b><extra></extra>"
    ))
    fig.update_layout(
        plot_bgcolor="rgba(17, 24, 39, 0.4)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F8FAFC", size=11),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(size=11, color="#CBD5E1")),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", title="Parts", tickfont=dict(size=10, color="#94A3B8")),
        margin=dict(l=45, r=20, t=25, b=35),
        height=300,
        showlegend=False,
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="rgba(255,255,255,0.15)", font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12))
    )
    return fig


# ---------------------------------------------------------------------------
# Load Intelligence Tables
# ---------------------------------------------------------------------------
part_intel = load_table("part_intelligence_summary")
customer_profiles = load_table("customer_profiles")
accuracy = load_table("part_forecast_accuracy")
demand_lines = load_table("demand_lines")
breakdown = load_table("part_customer_breakdown")
cust_yearly = load_table("customer_yearly_trend")
part_yearly = load_table("part_yearly_trend")
monthly_demand = load_table("part_monthly_demand")
tuned_params = load_table("part_tuned_parameters")
forecast_horizon = load_table("part_forecast_horizon")
risk_alerts = load_table("risk_alerts")
project_intel = load_table("project_intelligence")
item_master = load_table("item_master")
part_master = load_table("part_master")
replenishment = load_table("replenishment_recommendations")
forecast_decomposition = load_table("customer_forecast_decomposition")
forecast_horizon_adjusted = load_table("part_forecast_horizon_adjusted")

if len(demand_lines) > 0:
    part_desc_lookup = demand_lines.drop_duplicates(["project", "part_code"])[["project", "part_code", "part_desc"]]
    part_intel = part_intel.merge(part_desc_lookup, on=["project", "part_code"], how="left")
else:
    part_desc_lookup = pd.DataFrame(columns=["project", "part_code", "part_desc"])

# Freshness & System Header
_legacy_ts = get_freshness()
_fresh = refresh_panel.freshness_info(fallback_timestamp=None if _legacy_ts == "no data yet" else _legacy_ts)
_fresh_color = "#F59E0B" if _fresh["stale"] else "#10B981"
_dot_color = "#F59E0B" if _fresh["stale"] else "#10B981"
_source_label = escape(str(_fresh.get("source") or "SQLite intelligence store"))

# Top Executive Bar
st.markdown(f"""
<div class="scp-topbar">
  <div class="scp-brand-block">
    <div class="scp-brand-icon">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="4" y="4" width="16" height="16" rx="2"/>
        <rect x="9" y="9" width="6" height="6"/>
        <line x1="9" y1="1" x2="9" y2="4"/>
        <line x1="15" y1="1" x2="15" y2="4"/>
        <line x1="9" y1="20" x2="9" y2="23"/>
        <line x1="15" y1="20" x2="15" y2="23"/>
        <line x1="20" y1="9" x2="23" y2="9"/>
        <line x1="20" y1="14" x2="23" y2="14"/>
        <line x1="1" y1="9" x2="4" y2="9"/>
        <line x1="1" y1="14" x2="4" y2="14"/>
      </svg>
    </div>
    <div class="scp-brand-text">
      <div style="display:flex; align-items:center;">
        <div class="scp-wordmark">SCI<span>IP</span></div>
        <span class="scp-badge-tag">ENTERPRISE COMMAND CENTER</span>
      </div>
      <div class="scp-subtitle">Semiconductor Supply Chain &amp; Inventory Intelligence Hub &middot; Live Operations Telemetry</div>
    </div>
  </div>
  <div class="scp-meta">
    <div class="scp-freshness-pill" style="color:{_fresh_color};">
      <div class="scp-pulse-dot" style="background:{_dot_color};"></div>
      {_fresh["label"]}
    </div>
    <div class="scp-source">&bull; SOURCE: {_source_label}</div>
  </div>
</div>
""", unsafe_allow_html=True)

refresh_panel.show_pending_notifications()

# Background Copilot Launcher
copilot.render_copilot(DB_PATH)

if len(part_intel) == 0:
    st.info("No data loaded yet - upload your latest Sales Register below to get started.")
    refresh_panel.render_refresh_panel(key_prefix="first_run")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar Telemetry & Information Hub
# ---------------------------------------------------------------------------
all_projects = sorted(part_intel["project"].dropna().unique().tolist())

with st.sidebar:
    st.markdown("""
    <div class="scp-sidebar-header">
      <div class="scp-sidebar-brand">
        <div style="width:36px; height:36px; border-radius:8px; background:linear-gradient(135deg,#06B6D4,#3B82F6); display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700;">⚡</div>
        <div>
          <div style="font-weight:700; font-size:16px; color:#F8FAFC;">SCIIP HUB</div>
          <div style="font-size:11px; color:#94A3B8;">Semiconductor v2.4</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="scp-section-label">⚡ System Telemetry</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="scp-sidebar-card">
      <div style="font-size:11px; color:#94A3B8; margin-bottom:4px;">DATABASE STATUS</div>
      <div style="font-size:13px; font-weight:600; color:#10B981; display:flex; align-items:center; gap:6px;">
        <span style="width:7px; height:7px; border-radius:50%; background:#10B981;"></span> SQLite Connected
      </div>
      <div style="margin-top:10px; font-size:11px; color:#94A3B8;">TOTAL CATALOG</div>
      <div style="font-size:14px; font-weight:600; color:#F8FAFC; font-family:'JetBrains Mono',monospace;">{len(part_intel):,} Unique Parts</div>
      <div style="margin-top:6px; font-size:11px; color:#94A3B8;">ACCOUNTS ON RECORD</div>
      <div style="font-size:14px; font-weight:600; color:#F8FAFC; font-family:'JetBrains Mono',monospace;">{len(customer_profiles):,} Customers</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="scp-section-label">📚 Demand Pattern Guide</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="scp-sidebar-card" style="font-size:11px; line-height:1.6;">
      <div style="margin-bottom:8px;"><span class="scp-pill scp-pill-smooth">Smooth</span> <span style="color:#94A3B8;">Regular cadence, low demand volatility</span></div>
      <div style="margin-bottom:8px;"><span class="scp-pill scp-pill-erratic">Erratic</span> <span style="color:#94A3B8;">Regular cadence, high volume variability</span></div>
      <div style="margin-bottom:8px;"><span class="scp-pill scp-pill-intermittent">Intermittent</span> <span style="color:#94A3B8;">Sporadic orders, low volume variance</span></div>
      <div><span class="scp-pill scp-pill-lumpy">Lumpy</span> <span style="color:#94A3B8;">Highly sporadic &amp; unpredictable sizes</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    active_prov = copilot.load_active_provider_config(DB_PATH)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="scp-section-label">🤖 AI Copilot Status</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="scp-sidebar-card">
      <div style="font-size:11px; color:#94A3B8; margin-bottom:4px;">ACTIVE PROVIDER</div>
      <div style="font-size:13px; font-weight:600; color:#38BDF8; display:flex; align-items:center; gap:6px;">
        <span style="width:7px; height:7px; border-radius:50%; background:#10B981;"></span> {active_prov.name}
      </div>
      <div style="font-size:11px; color:#64748B; margin-top:3px;">Model: {active_prov.model}</div>
      <div style="font-size:11px; color:#94A3B8; margin-top:6px;">&bull; Full chat &amp; history in the <b>AI Copilot</b> tab</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Scope & Executive Operational Snapshot
# ---------------------------------------------------------------------------
st.markdown('<div class="scp-section-label">🎯 Operating Scope &middot; Program Filter</div>', unsafe_allow_html=True)
project_filter = st.radio("Project", ["All"] + all_projects, horizontal=True, label_visibility="collapsed")

def scope(df, col="project"):
    if project_filter == "All" or col not in df.columns:
        return df
    return df[df[col] == project_filter]

part_intel_f = scope(part_intel)
breakdown_f = scope(breakdown)
demand_lines_f = scope(demand_lines)
part_yearly_f = scope(part_yearly)
monthly_demand_f = scope(monthly_demand)
accuracy_f = scope(accuracy)
risk_alerts_f = scope(risk_alerts)
forecast_horizon_f = scope(forecast_horizon)
tuned_params_f = scope(tuned_params)

customers_in_scope = customer_profiles["cust_name"] if project_filter == "All" else breakdown_f["cust_name"].unique()
customer_profiles_f = customer_profiles[customer_profiles["cust_name"].isin(customers_in_scope)]

# SVG icons for KPI Cards
ICON_CHIP = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>'
ICON_USERS = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'
ICON_WAVE = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'
ICON_TARGET = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>'
ICON_MONEY = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>'

st.markdown("""
<div class="scp-snapshot-header">
  <div class="scp-overview-title">Operational Snapshot</div>
  <div class="scp-overview-subtitle">Real-time demand, customer health, forecast accuracy, and order exposure across the selected scope.</div>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(render_kpi("Active Parts", f"{len(part_intel_f):,}", f"&bull; Program: {project_filter}", "#06B6D4", ICON_CHIP), unsafe_allow_html=True)
with k2:
    st.markdown(render_kpi("Accounts / Customers", f"{len(customer_profiles_f):,}", "&bull; Active customer accounts", "#6366F1", ICON_USERS), unsafe_allow_html=True)
with k3:
    pc = part_intel_f["demand_pattern"].value_counts().to_dict() if len(part_intel_f) else {}
    dom = max(pc, key=pc.get) if pc else "-"
    lp = 100 * part_intel_f["demand_pattern"].isin(["Lumpy", "Erratic"]).mean() if len(part_intel_f) else 0
    dom_color = PATTERN_COLOR.get(dom, "#10B981")
    st.markdown(render_kpi("Dominant Pattern", dom, f"&bull; {lp:.0f}% Lumpy / Erratic", dom_color, ICON_WAVE), unsafe_allow_html=True)
with k4:
    if len(accuracy_f) > 0:
        hit_rate = f"{accuracy_f['occurrence_hit_rate_pct'].mean():.0f}%"
        sub = "&bull; Backtested occurrence"
    else:
        hit_rate = "-"
        sub = "&bull; No backtest data"
    st.markdown(render_kpi("Avg Forecast Hit Rate", hit_rate, sub, "#F59E0B", ICON_TARGET), unsafe_allow_html=True)
with k5:
    if len(demand_lines_f) > 0:
        opn = demand_lines_f[demand_lines_f["status"] == "OPEN"]
        ov = (opn["open_qty"] * opn["unit_price"]).sum()
        ov_str = f"${ov:,.0f}"
        open_lines = len(opn)
        sub = f"&bull; {open_lines:,} open order lines"
    else:
        ov_str = "-"
        sub = "&bull; Zero open orders"
    st.markdown(render_kpi("Open Order Value", ov_str, sub, "#10B981", ICON_MONEY), unsafe_allow_html=True)

st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Intelligence Modules Navigation
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "📊  Overview",
    "👥  Customer Intelligence",
    "🔬  Part Intelligence",
    "💳  Transaction Intelligence",
    "📈  Forecast Intelligence",
    "📦  Replenishment",
    "📋  Item Master",
    "🤖  AI Copilot",
    "🔄  Data Refresh"
])
tab_overview, tab_customer, tab_part, tab_txn, tab_forecast, tab_replen, tab_item, tab_copilot, tab_refresh = tabs

# ===========================================================================
# TAB 1: OVERVIEW
# ===========================================================================
with tab_overview:
    st.markdown("<div class='scp-overview-title'>Network Overview</div><div class='scp-overview-subtitle'>Pattern mix, alert concentration, project performance, and relationship watchlist.</div>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        with st.container(border=True):
            st.markdown('<div class="scp-section-label">📊 Demand Pattern Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(pattern_distribution_chart(part_intel_f), use_container_width=True, config={"displayModeBar": False}, key="overview_pattern_distribution")
    with cb:
        with st.container(border=True):
            st.markdown('<div class="scp-section-label">⚠️ Risk &amp; Alerts Summary</div>', unsafe_allow_html=True)
            if len(risk_alerts_f) > 0:
                counts = risk_alerts_f["alert_type"].value_counts()
                fig = go.Figure(go.Bar(
                    x=counts.values,
                    y=counts.index,
                    orientation="h",
                    marker=dict(color="#F59E0B", line=dict(color="#D97706", width=1)),
                    text=counts.values,
                    textposition="outside",
                    textfont=dict(color="#F8FAFC", family="Space Grotesk, sans-serif", size=11),
                    hovertemplate="<b>%{y}</b><br>Alert Count: <b>%{x}</b><extra></extra>"
                ))
                fig.update_layout(
                    plot_bgcolor="rgba(17, 24, 39, 0.4)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#F8FAFC", size=11),
                    xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(size=10, color="#94A3B8")),
                    yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(size=11, color="#CBD5E1")),
                    margin=dict(l=10, r=30, t=15, b=30),
                    height=300,
                    hoverlabel=dict(bgcolor="#0F172A", bordercolor="rgba(255,255,255,0.15)", font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12))
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="overview_risk_alerts")
            else:
                st.caption("No alerts in scope.")

    cc, cd = st.columns(2)
    with cc:
        with st.container(border=True):
            st.markdown('<div class="scp-section-label">🚀 Project Intelligence</div>', unsafe_allow_html=True)
            if len(project_intel) > 0:
                pi_show = project_intel[["project", "total_qty", "total_revenue", "num_customers", "num_parts", "yoy_growth_pct", "overdue_relationships_count"]]
                st.dataframe(pi_show, use_container_width=True, hide_index=True, height=270)
            else:
                st.caption("No project intelligence records.")
    with cd:
        with st.container(border=True):
            st.markdown('<div class="scp-section-label">⏱️ Relationship Watchlist (Overdue)</div>', unsafe_allow_html=True)
            overdue = breakdown_f[breakdown_f["relationship_status"] == "Overdue / went quiet"].sort_values("days_since_last_order", ascending=False).head(8)
            if len(overdue) > 0:
                for _, r in overdue.iterrows():
                    st.markdown(f"""
                    <div class="scp-row-item">
                      <span style="font-weight:500;">{r["cust_name"]} &middot; <span class="scp-mono" style="color:#38BDF8;">{r["part_code"]}</span></span>
                      <span class="scp-pill scp-pill-lumpy">{int(r["days_since_last_order"])}d overdue</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No overdue relationships in scope.")

# ===========================================================================
# TAB 2: CUSTOMER INTELLIGENCE
# ===========================================================================
with tab_customer:
    col_list, col_detail = st.columns([1, 1.6])
    with col_list:
        st.markdown('<div class="scp-section-label">📁 Customer Directory</div>', unsafe_allow_html=True)
        search = st.text_input("Search customer", label_visibility="collapsed", placeholder="🔍 Search customer name...", key="cust_search")
        cp = customer_profiles_f.copy().reset_index(drop=True)
        if search:
            cp = cp[cp["cust_name"].str.contains(search, case=False, na=False)].reset_index(drop=True)
        cp_display = cp[["cust_name", "total_revenue", "recency_days", "currently_open_lines"]].rename(
            columns={"cust_name": "Customer", "total_revenue": "Revenue ($)", "recency_days": "Days Since Order", "currently_open_lines": "Open Lines"}
        )
        st.markdown('<div class="scp-hint">💡 Click any customer row to inspect their executive dossier</div>', unsafe_allow_html=True)
        ev = st.dataframe(cp_display, use_container_width=True, height=420, hide_index=True, on_select="rerun", selection_mode="single-row", key="cust_table")
        selected_customer = cp.iloc[ev.selection.rows[0]]["cust_name"] if ev.selection.rows else None

    with col_detail:
        if selected_customer:
            prof = customer_profiles[customer_profiles["cust_name"] == selected_customer].iloc[0]
            st.markdown('<div class="scp-panel">', unsafe_allow_html=True)
            
            # Customer Executive Dossier Header
            initials = "".join([w[0] for w in str(selected_customer).split()[:2]]).upper() or "CU"
            st.markdown(f"""
            <div class="scp-dossier-header">
              <div class="scp-avatar">{initials}</div>
              <div>
                <div class="scp-dossier-title">{selected_customer}</div>
                <div class="scp-dossier-badges">
                  <span class="scp-pill scp-pill-active">ACTIVE ACCOUNT</span>
                  <span class="scp-meta-chip">Scope: {project_filter}</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(render_kpi("Total Revenue", f"${prof['total_revenue']:,.0f}", "&bull; Historical value", "#10B981"), unsafe_allow_html=True)
            with c2:
                gap_val = f"{prof['avg_order_interval_days']} d" if pd.notna(prof['avg_order_interval_days']) else "-"
                st.markdown(render_kpi("Avg Order Gap", gap_val, "&bull; Reorder frequency", "#06B6D4"), unsafe_allow_html=True)
            with c3:
                st.markdown(render_kpi("Parts Ordered", f"{prof['distinct_parts']}", "&bull; Catalog breadth", "#6366F1"), unsafe_allow_html=True)
            with c4:
                st.markdown(render_kpi("Open Backlog Qty", f"{prof['currently_open_qty']:,.0f}", "&bull; Pending fulfillment", "#F59E0B"), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="scp-section-label">🔬 Parts Ordered by this Customer &middot; Select a Part for Monthly Cadence</div>', unsafe_allow_html=True)
            cust_parts = breakdown_f[breakdown_f["cust_name"] == selected_customer].sort_values("cust_qty", ascending=False).reset_index(drop=True)
            desc_lookup = demand_lines.drop_duplicates(["project", "part_code"])[["project", "part_code", "part_desc"]]
            cust_parts = cust_parts.merge(desc_lookup, on=["project", "part_code"], how="left")
            cp_show = cust_parts[["part_code", "part_desc", "project", "cust_qty", "relationship_status"]].rename(
                columns={"part_code": "Part", "part_desc": "Description", "project": "Project", "cust_qty": "Total Qty", "relationship_status": "Status"}
            )
            ev2 = st.dataframe(cp_show, use_container_width=True, hide_index=True, height=210, on_select="rerun", selection_mode="single-row", key="cust_parts_table")
            if ev2.selection.rows:
                sel = cust_parts.iloc[ev2.selection.rows[0]]
                hist = demand_lines[(demand_lines["part_code"] == sel["part_code"]) & (demand_lines["project"] == sel["project"]) & (demand_lines["cust_name"] == selected_customer)].copy()
                hist["order_month"] = pd.to_datetime(hist["so_date"], errors="coerce").dt.to_period("M")
                hist = hist.dropna(subset=["order_month"])
                if len(hist) > 0:
                    monthly_qty = hist.groupby("order_month")["original_qty"].sum()
                    calendar_months = pd.period_range(monthly_qty.index.min(), monthly_qty.index.max(), freq="M")
                    by_month = monthly_qty.reindex(calendar_months, fill_value=0).rename("qty").reset_index()
                    by_month = by_month.rename(columns={"index": "period"})
                    by_month["month"] = by_month["period"].astype(str)
                    by_month["mom_pct"] = by_month["qty"].pct_change().replace([np.inf, -np.inf], np.nan) * 100
                    latest = by_month.iloc[-1]
                    active_months = int((by_month["qty"] > 0).sum())
                    avg_monthly = by_month["qty"].mean()
                    mom_text = "New month" if pd.isna(latest["mom_pct"]) and latest["qty"] > 0 else (f'{latest["mom_pct"]:+.1f}%' if pd.notna(latest["mom_pct"]) else "-")

                    st.markdown(f'<div class="scp-section-label">📈 {sel["part_code"]} &mdash; Monthly Customer Demand Profile</div>', unsafe_allow_html=True)
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(render_kpi("Total Qty", f"{by_month['qty'].sum():,.0f}", "&bull; Cumulative", "#06B6D4"), unsafe_allow_html=True)
                    with m2:
                        st.markdown(render_kpi("Active Months", f"{active_months}", "&bull; Months with orders", "#10B981"), unsafe_allow_html=True)
                    with m3:
                        st.markdown(render_kpi("Avg Monthly Qty", f"{avg_monthly:,.1f}", "&bull; Mean run-rate", "#6366F1"), unsafe_allow_html=True)
                    with m4:
                        st.markdown(render_kpi("Latest MoM", f"{mom_text}", "&bull; Month-over-month", "#F59E0B"), unsafe_allow_html=True)

                    fig = go.Figure(go.Bar(
                        x=by_month["month"],
                        y=by_month["qty"],
                        marker=dict(color="#06B6D4", line=dict(color="#0891B2", width=1)),
                        customdata=by_month["mom_pct"].tolist(),
                        hovertemplate="<b>%{x}</b><br>Quantity: <b>%{y:,.0f}</b> units<br>MoM Change: <b>%{customdata:+.1f}%</b><extra></extra>"
                    ))
                    fig.update_layout(
                        plot_bgcolor="rgba(17, 24, 39, 0.4)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color="#F8FAFC", size=11),
                        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", title="Order month"),
                        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", title="Quantity"),
                        margin=dict(l=45, r=20, t=18, b=40),
                        height=260,
                        showlegend=False,
                        hoverlabel=dict(bgcolor="#0F172A", bordercolor="rgba(255,255,255,0.15)", font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12))
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"customer_part_monthly_{selected_customer}_{sel['part_code']}_{sel['project']}")

                    st.markdown('<div class="scp-section-label">📜 Sales Order History &middot; Selected Customer &amp; Part</div>', unsafe_allow_html=True)
                    so_cols = [c for c in ["so_no", "so_date", "original_qty", "shipped_qty", "open_qty", "status"] if c in hist.columns]
                    so_history = hist[so_cols].sort_values("so_date", ascending=False).rename(columns={
                        "so_no": "SO No.", "so_date": "Date", "original_qty": "Ordered",
                        "shipped_qty": "Shipped", "open_qty": "Open", "status": "Status"
                    })
                    st.dataframe(so_history, use_container_width=True, hide_index=True, height=220)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="scp-section-label">📦 Recent Order History (Last 10 Sales Orders)</div>', unsafe_allow_html=True)
            oh = demand_lines_f[demand_lines_f["cust_name"] == selected_customer].sort_values("so_date", ascending=False).head(10)
            if len(oh) > 0:
                st.dataframe(oh[["so_no", "so_date", "part_code", "original_qty", "status"]].rename(
                    columns={"so_no": "SO No.", "so_date": "Date", "part_code": "Part Code", "original_qty": "Quantity", "status": "Status"}),
                    use_container_width=True, hide_index=True, height=230)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="scp-section-label">💰 Annual Revenue Trajectory</div>', unsafe_allow_html=True)
            cy = cust_yearly[cust_yearly["cust_name"] == selected_customer]
            if project_filter != "All":
                cy = cy[cy["project"] == project_filter]
            if len(cy) > 0:
                yc = sorted([c for c in cy.columns if c.isdigit()])
                for _, r in cy.iterrows():
                    st.plotly_chart(yearly_bar_chart(r, yc, title=f"Program: {r['project']}"), use_container_width=True, config={"displayModeBar": False}, key=f"customer_revenue_yearly_{selected_customer}_{r['project']}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="scp-empty-state">
              <div class="scp-empty-icon">👥</div>
              <div class="scp-empty-title">Select an Account to Inspect</div>
              <div class="scp-empty-desc">Click any customer row on the directory table to review order cadence, purchased parts, monthly demand trends, and annual revenue trajectory.</div>
            </div>
            """, unsafe_allow_html=True)

# ===========================================================================
# TAB 3: PART INTELLIGENCE
# ===========================================================================
with tab_part:
    col_list, col_detail = st.columns([1, 1.6])
    with col_list:
        st.markdown('<div class="scp-section-label">🔬 Part Catalog</div>', unsafe_allow_html=True)
        search_p = st.text_input("Search parts", label_visibility="collapsed", placeholder="🔍 Search part code or description...", key="part_search")
        pi = part_intel_f.copy().reset_index(drop=True)
        if search_p:
            search_terms = [term.strip() for term in search_p.replace("\n", ",").split(",") if term.strip()]
            mask = pd.Series(False, index=pi.index)
            for term in search_terms:
                mask = mask | pi["part_code"].str.contains(term, case=False, na=False, regex=False)
                if "part_desc" in pi.columns:
                    mask = mask | pi["part_desc"].str.contains(term, case=False, na=False, regex=False)
            pi = pi[mask].reset_index(drop=True)
        pi_display = pi[["part_code", "part_desc", "project", "demand_pattern", "forecast_monthly_qty", "top_customer"]].rename(
            columns={"part_code": "Part Code", "part_desc": "Description", "project": "Project", "demand_pattern": "Pattern", "forecast_monthly_qty": "Fcst/Mo", "top_customer": "Top Customer"}
        )
        st.markdown('<div class="scp-hint">💡 Click 1 part for full dossier, or select multiple to compare side-by-side</div>', unsafe_allow_html=True)
        pev = st.dataframe(pi_display, use_container_width=True, height=420, hide_index=True, on_select="rerun", selection_mode="multi-row", key="part_table")
        sel_idx = pev.selection.rows

    with col_detail:
        if not sel_idx:
            st.markdown("""
            <div class="scp-empty-state">
              <div class="scp-empty-icon">🔬</div>
              <div class="scp-empty-title">Select Part(s) to Analyze</div>
              <div class="scp-empty-desc">Click a single part to view its full demand profile, pricing dynamics, and customer decomposition &mdash; or select multiple parts to compare them side-by-side.</div>
            </div>
            """, unsafe_allow_html=True)
        elif len(sel_idx) == 1:
            sel = pi.iloc[sel_idx[0]]
            row = part_intel[(part_intel["part_code"] == sel["part_code"]) & (part_intel["project"] == sel["project"])].iloc[0]
            st.markdown('<div class="scp-panel">', unsafe_allow_html=True)
            
            # Part Dossier Header
            p_desc = escape(str(row["part_desc"])) if pd.notna(row.get("part_desc")) else "No description specified"
            st.markdown(f"""
            <div class="scp-dossier-header">
              <div class="scp-avatar scp-avatar-part">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/></svg>
              </div>
              <div>
                <div class="scp-dossier-title">{sel['part_code']}</div>
                <div class="scp-desc" style="margin-bottom:6px;">{p_desc}</div>
                <div class="scp-dossier-badges">
                  {pattern_pill(row['demand_pattern'])}
                  <span class="scp-meta-chip">Program: {sel['project']}</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(render_kpi("Forecast / Month", f"{row['forecast_monthly_qty']:.1f}", "&bull; Statistical model", "#06B6D4"), unsafe_allow_html=True)
            with c2:
                occ = row.get("occurrence_probability_this_month")
                occ_str = f"{occ:.2f}" if isinstance(occ, (int, float)) and pd.notna(occ) else str(occ if pd.notna(occ) else "-")
                st.markdown(render_kpi("Occurrence Prob.", occ_str, "&bull; Demand likelihood", "#10B981"), unsafe_allow_html=True)
            with c3:
                st.markdown(render_kpi("Top Customer", f"{row['top_customer']}", "&bull; Largest account", "#6366F1"), unsafe_allow_html=True)
            with c4:
                st.markdown(render_kpi("Concentration", f"{row['top_customer_pct_of_demand']:.0f}%", "&bull; Share of demand", "#F59E0B"), unsafe_allow_html=True)

            if len(forecast_decomposition) > 0:
                fd = forecast_decomposition[(forecast_decomposition["part_code"] == sel["part_code"]) & (forecast_decomposition["project"] == sel["project"])].sort_values("forecast_contribution", ascending=False)
                if len(fd) > 0:
                    st.markdown('<div class="scp-subsection"><div class="scp-section-label">👥 Forecast Decomposition &middot; Customer Drivers</div></div>', unsafe_allow_html=True)
                    st.markdown('<div class="scp-hint">Aggregate forecast split by each customer\'s historical share of this part\'s demand. Overdue customers are flagged.</div>', unsafe_allow_html=True)
                    fd_show = fd[["cust_name", "historical_pct_of_demand", "forecast_contribution", "relationship_status", "days_since_last_order"]].rename(
                        columns={"cust_name": "Customer", "historical_pct_of_demand": "Hist. Share %", "forecast_contribution": "Fcst Contribution", "relationship_status": "Status", "days_since_last_order": "Days Since Order"}
                    )
                    st.dataframe(fd_show, use_container_width=True, hide_index=True, height=min(220, 48 + 38 * len(fd_show)))

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="scp-section-label">📈 Monthly Demand Cadence</div>', unsafe_allow_html=True)
            pm = monthly_demand[(monthly_demand["part_code"] == sel["part_code"]) & (monthly_demand["project"] == sel["project"])].sort_values("month")
            if len(pm) > 0:
                st.plotly_chart(waveform_chart(pm, title="Aggregated Monthly Demand Waveform"), use_container_width=True, config={"displayModeBar": False}, key=f"part_monthly_demand_{sel['part_code']}_{sel['project']}")

            py = part_yearly[(part_yearly["part_code"] == sel["part_code"]) & (part_yearly["project"] == sel["project"])]
            if len(py) > 0:
                yc = sorted([c for c in py.columns if c.isdigit()])
                st.markdown('<div class="scp-section-label">📅 Annual Quantity Trend</div>', unsafe_allow_html=True)
                st.plotly_chart(yearly_bar_chart(py.iloc[0], yc, title="Year-over-Year Ordered Volume"), use_container_width=True, config={"displayModeBar": False}, key=f"part_yearly_trend_{sel['part_code']}_{sel['project']}")

            pc = breakdown[(breakdown["part_code"] == sel["part_code"]) & (breakdown["project"] == sel["project"])].sort_values("cust_qty", ascending=False).reset_index(drop=True)
            if "unit_price" in demand_lines.columns and len(demand_lines) > 0:
                st.markdown('<div class="scp-subsection"><div class="scp-section-label">🏷️ Pricing &amp; Customer Movement Analysis</div></div>', unsafe_allow_html=True)
                price_history = demand_lines[(demand_lines["part_code"] == sel["part_code"]) & (demand_lines["project"] == sel["project"])].copy()
                price_history["order_date"] = pd.to_datetime(price_history["so_date"], errors="coerce")
                price_history["unit_price"] = pd.to_numeric(price_history["unit_price"], errors="coerce")
                price_history = price_history.dropna(subset=["order_date", "unit_price"])
                if len(price_history) > 0:
                    price_history = price_history.sort_values("order_date")
                    pricing_rows = []
                    for customer, customer_orders in price_history.groupby("cust_name"):
                        first_order = customer_orders.iloc[0]
                        latest_order = customer_orders.iloc[-1]
                        first_price = float(first_order["unit_price"])
                        latest_price = float(latest_order["unit_price"])
                        change_pct = None if first_price == 0 else round((latest_price - first_price) / first_price * 100, 1)
                        relationship = pc.loc[pc["cust_name"] == customer, "relationship_status"]
                        pricing_rows.append({
                            "Customer": customer,
                            "Earlier Price": first_price,
                            "Latest Price": latest_price,
                            "Change": change_pct,
                            "First Order": first_order["order_date"].date(),
                            "Last Order": latest_order["order_date"].date(),
                            "Orders": len(customer_orders),
                            "Ordering Status": relationship.iloc[0] if len(relationship) else "-",
                        })
                    pricing_df = pd.DataFrame(pricing_rows).sort_values("Last Order", ascending=False)
                    price_changes = pricing_df["Change"].dropna()
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.markdown(render_kpi("Accounts Analyzed", f"{len(pricing_df)}", "&bull; Buying this part", "#06B6D4"), unsafe_allow_html=True)
                    with p2:
                        st.markdown(render_kpi("Price Movements", f"{int((price_changes.abs() > 0.1).sum())}", "&bull; Variance detected", "#F59E0B"), unsafe_allow_html=True)
                    with p3:
                        st.markdown(render_kpi("Churned / Quiet", f"{int((pricing_df['Ordering Status'] == 'Overdue / went quiet').sum())}", "&bull; Overdue accounts", "#EF4444"), unsafe_allow_html=True)
                    
                    price_history["original_qty"] = pd.to_numeric(price_history["original_qty"], errors="coerce").fillna(0)
                    price_history["price_value"] = price_history["unit_price"] * price_history["original_qty"]
                    monthly_price = price_history.groupby(price_history["order_date"].dt.to_period("M"))[["price_value", "original_qty"]].sum()
                    monthly_price["weighted_price"] = monthly_price["price_value"].div(monthly_price["original_qty"].replace(0, np.nan))
                    monthly_price = monthly_price.reset_index().rename(columns={"order_date": "period"})
                    monthly_price["month"] = monthly_price["period"].astype(str)
                    
                    price_fig = go.Figure(go.Scatter(
                        x=monthly_price["month"],
                        y=monthly_price["weighted_price"],
                        mode="lines+markers",
                        line=dict(color="#10B981", width=2.5),
                        marker=dict(size=7, color="#059669"),
                        hovertemplate="<b>%{x}</b><br>Weighted Avg Price: <b>$%{y:,.2f}</b><extra></extra>"
                    ))
                    price_fig.update_layout(
                        title=dict(text="Weighted Average Selling Price Trajectory", font=dict(family="Space Grotesk, sans-serif", size=12, color="#94A3B8")),
                        plot_bgcolor="rgba(17, 24, 39, 0.4)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color="#F8FAFC", size=11),
                        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", title="Order month"),
                        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", title="Unit Price ($)"),
                        margin=dict(l=45, r=20, t=32, b=40),
                        height=240,
                        showlegend=False,
                        hoverlabel=dict(bgcolor="#0F172A", bordercolor="rgba(255,255,255,0.15)", font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12))
                    )
                    st.plotly_chart(price_fig, use_container_width=True, config={"displayModeBar": False}, key=f"part_pricing_history_{sel['part_code']}_{sel['project']}")
                    
                    pricing_display = pricing_df.copy()
                    pricing_display["Earlier Price"] = pricing_display["Earlier Price"].map(lambda v: f"${v:,.2f}")
                    pricing_display["Latest Price"] = pricing_display["Latest Price"].map(lambda v: f"${v:,.2f}")
                    pricing_display["Change"] = pricing_display["Change"].map(lambda v: "-" if pd.isna(v) else f"{v:+.1f}%")
                    pricing_display["Ordering Status"] = pricing_display["Ordering Status"].map(lambda v: "STOPPED / WENT QUIET" if v == "Overdue / went quiet" else v)
                    st.dataframe(pricing_display, use_container_width=True, hide_index=True, height=240)

            st.markdown('<div class="scp-subsection"><div class="scp-section-label">👥 Customer Breakdown &middot; Who is Active vs. Quiet</div></div>', unsafe_allow_html=True)
            pc_show = pc[["cust_name", "cust_qty", "pct_of_part_demand", "relationship_status", "days_since_last_order"]].rename(
                columns={"cust_name": "Customer", "cust_qty": "Total Qty", "pct_of_part_demand": "% Share", "relationship_status": "Status", "days_since_last_order": "Days Since Order"}
            )
            cev = st.dataframe(pc_show, use_container_width=True, hide_index=True, height=180, on_select="rerun", selection_mode="single-row", key="part_cust_table")
            if cev.selection.rows:
                scust = pc.iloc[cev.selection.rows[0]]["cust_name"]
                ch = demand_lines[(demand_lines["part_code"] == sel["part_code"]) & (demand_lines["project"] == sel["project"]) & (demand_lines["cust_name"] == scust)].copy()
                ch["order_month"] = pd.to_datetime(ch["so_date"], errors="coerce").dt.to_period("M")
                ch = ch.dropna(subset=["order_month"])
                if len(ch) > 0:
                    monthly_qty = ch.groupby("order_month")["original_qty"].sum()
                    calendar_months = pd.period_range(monthly_qty.index.min(), monthly_qty.index.max(), freq="M")
                    by_month = monthly_qty.reindex(calendar_months, fill_value=0).rename("qty").reset_index()
                    by_month = by_month.rename(columns={"index": "period"})
                    by_month["month"] = by_month["period"].astype(str)
                    st.markdown(f'<div class="scp-section-label">📊 {scust} &mdash; Monthly Demand for {sel["part_code"]}</div>', unsafe_allow_html=True)
                    st.plotly_chart(waveform_chart(by_month), use_container_width=True, config={"displayModeBar": False}, key=f"part_customer_monthly_{sel['part_code']}_{sel['project']}_{scust}")
                st.markdown('<div class="scp-hint">Underlying transactions for this customer + part:</div>', unsafe_allow_html=True)
                st.dataframe(ch[["so_no", "so_date", "original_qty", "shipped_qty", "status"]].rename(
                    columns={"so_no": "SO No.", "so_date": "Date", "original_qty": "Qty", "shipped_qty": "Shipped", "status": "Status"}),
                    use_container_width=True, hide_index=True)

            tp = tuned_params[(tuned_params["part_code"] == sel["part_code"]) & (tuned_params["project"] == sel["project"])]
            acc = accuracy[(accuracy["part_code"] == sel["part_code"]) & (accuracy["project"] == sel["project"])]
            if len(tp) > 0 or len(acc) > 0:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="scp-section-label">🎯 Forecast Diagnostics &amp; Model Parameters</div>', unsafe_allow_html=True)
                if len(acc) > 0:
                    a = acc.iloc[0]
                    st.caption(f"&bull; Backtest Evaluation: {a['occurrence_hit_rate_pct']}% occurrence hit rate over {a['num_test_points']} test months (average magnitude error {a['avg_magnitude_error_pct']}%)")
                if len(tp) > 0:
                    t = tp.iloc[0]
                    st.caption(f"&bull; Croston/SBA Tuned Parameters: alpha={t['tuned_alpha']}, beta={t['tuned_beta']}, threshold={t['tuned_threshold']} ({'Meets accuracy baseline' if t['met_baseline'] else 'Below standard baseline'})")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            sel_rows = pi.iloc[sel_idx]
            st.markdown('<div class="scp-panel">', unsafe_allow_html=True)
            st.markdown(f"### 🔬 Side-by-Side Comparison: {len(sel_rows)} Selected Parts")
            compare_rows, series = [], []
            for _, sel in sel_rows.iterrows():
                fr = part_intel[(part_intel["part_code"] == sel["part_code"]) & (part_intel["project"] == sel["project"])].iloc[0]
                compare_rows.append({
                    "Part": sel["part_code"], "Project": sel["project"], "Pattern": fr["demand_pattern"],
                    "Fcst/Mo": fr["forecast_monthly_qty"], "Top Customer": fr["top_customer"],
                    "Concentration": f'{fr["top_customer_pct_of_demand"]:.0f}%', "Total Qty": fr["total_qty"]
                })
                m = monthly_demand[(monthly_demand["part_code"] == sel["part_code"]) & (monthly_demand["project"] == sel["project"])].sort_values("month")
                if len(m) > 0:
                    series.append((f'{sel["part_code"]} ({sel["project"]})', m))
            st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if series:
                st.plotly_chart(multi_waveform_chart(series), use_container_width=True, config={"displayModeBar": False}, key="part_compare_monthly_demand")
            st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# TAB 4: TRANSACTION INTELLIGENCE
# ===========================================================================
with tab_txn:
    st.markdown('<div class="scp-panel">', unsafe_allow_html=True)
    st.markdown('<div class="scp-section-label">⚡ Filter Transaction Records</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)
    txn_status = fc1.selectbox("Status", ["All", "OPEN", "CLOSED"])
    txn_cust = fc2.selectbox("Customer", ["All"] + sorted(demand_lines_f["cust_name"].dropna().unique().tolist()))
    txn_review = fc4.selectbox("Needs review only", ["No", "Yes"])

    txn_parts_avail = demand_lines_f.drop_duplicates(["project", "part_code"])[["project", "part_code", "part_desc"]].copy()
    txn_parts_avail["_label"] = txn_parts_avail["part_code"] + " - " + txn_parts_avail["part_desc"].fillna("").str.slice(0, 50)
    with fc3:
        txn_part_labels = st.multiselect("Part(s)", txn_parts_avail["_label"].tolist(), label_visibility="visible",
                                           placeholder="Search part code or description...")

    txn = demand_lines_f.copy()
    if txn_status != "All":
        txn = txn[txn["status"] == txn_status]
    if txn_cust != "All":
        txn = txn[txn["cust_name"] == txn_cust]
    if txn_part_labels:
        sel_parts = txn_parts_avail[txn_parts_avail["_label"].isin(txn_part_labels)]["part_code"].tolist()
        txn = txn[txn["part_code"].isin(sel_parts)]
    if txn_review == "Yes":
        txn = txn[txn["needs_review"] == True]

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown(render_kpi("Lines in View", f"{len(txn):,}", "&bull; Filtered records", "#06B6D4"), unsafe_allow_html=True)
    with t2:
        st.markdown(render_kpi("Open Backlog Qty", f"{txn[txn.status == 'OPEN']['open_qty'].sum():,.0f}", "&bull; Unshipped volume", "#F59E0B"), unsafe_allow_html=True)
    with t3:
        st.markdown(render_kpi("Shipped Volume", f"{txn['shipped_qty'].sum():,.0f}", "&bull; Fulfilled units", "#10B981"), unsafe_allow_html=True)
    with t4:
        st.markdown(render_kpi("Needs Review", f"{int(txn['needs_review'].sum())}", "&bull; Attention flagged", "#EF4444"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    cols = ["so_no", "so_date", "cust_name", "part_code", "part_desc", "project", "original_qty", "shipped_qty", "open_qty",
            "status", "num_shipments", "fulfilling_dos", "actual_days_to_close", "quoted_lead_time_days", "needs_review"]
    cols = [c for c in cols if c in txn.columns]
    st.dataframe(txn[cols].sort_values("so_date", ascending=False).rename(columns={"part_desc": "Description"}),
                 use_container_width=True, hide_index=True, height=520)
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# TAB 5: FORECAST INTELLIGENCE
# ===========================================================================
with tab_forecast:
    st.markdown('<div class="scp-panel">', unsafe_allow_html=True)
    st.markdown('<div class="scp-section-label">🔮 6-Month Rolling Forecast Horizon &middot; Search &amp; Inspect Parts</div>', unsafe_allow_html=True)
    fh_parts_avail = part_intel_f.drop_duplicates(["project", "part_code"])[["project", "part_code", "part_desc"]].copy() if len(part_intel_f) else pd.DataFrame(columns=["project", "part_code", "part_desc"])
    fh_parts_avail["_label"] = fh_parts_avail["part_code"] + " - " + fh_parts_avail["part_desc"].fillna("").str.slice(0, 50)
    fh_selected = st.multiselect("Parts for forecast detail", fh_parts_avail["_label"].tolist(), label_visibility="collapsed",
                                   placeholder="🔍 Search part code or description for rolling forecast...")
    fh = forecast_horizon_f.copy()
    if fh_selected:
        sel_parts = fh_parts_avail[fh_parts_avail["_label"].isin(fh_selected)]["part_code"].tolist()
        fh = fh[fh["part_code"].isin(sel_parts)]
    if len(fh) > 0:
        fh = fh.merge(part_desc_lookup, on=["project", "part_code"], how="left")
        fh_cols = ["project", "part_code", "part_desc", "forecast_month", "predicted_to_occur", "occurrence_probability", "forecast_qty", "cumulative_qty"]
        st.dataframe(fh[fh_cols].rename(columns={"part_desc": "Description"}), use_container_width=True, hide_index=True, height=300)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="scp-section-label">📊 All Part Forecasts &middot; Statistical Baselines</div>', unsafe_allow_html=True)
    fc_cols = ["part_code", "part_desc", "project", "demand_pattern", "forecast_method", "occurrence_probability_this_month",
               "predicted_to_occur_this_month", "forecast_monthly_qty"]
    fc_cols = [c for c in fc_cols if c in part_intel_f.columns]
    st.dataframe(part_intel_f[fc_cols].rename(columns={"part_desc": "Description"}), use_container_width=True, hide_index=True, height=350)

    if len(forecast_horizon_adjusted) > 0:
        adjusted_only = forecast_horizon_adjusted[forecast_horizon_adjusted["adjustment_note"].notna()]
        if len(adjusted_only) > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="scp-section-label">🤝 Customer-Adjusted Forecasts &middot; Real Commitments Applied</div>', unsafe_allow_html=True)
            st.markdown('<div class="scp-hint">Statistical forecast replaced with real customer-stated quantities where provided &mdash; uncommitted demand untouched.</div>', unsafe_allow_html=True)
            adj_show = adjusted_only[["project", "part_code", "forecast_month", "forecast_qty", "adjusted_forecast_qty", "adjustment_note"]].rename(
                columns={"forecast_qty": "Statistical Fcst", "adjusted_forecast_qty": "Adjusted Fcst", "adjustment_note": "Reason / Note"}
            )
            st.dataframe(adj_show, use_container_width=True, hide_index=True, height=220)
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# TAB 6: REPLENISHMENT
# ===========================================================================
with tab_replen:
    st.markdown('<div class="scp-panel">', unsafe_allow_html=True)
    if len(replenishment) == 0:
        st.info("No replenishment data available yet &mdash; upload an Item Master file in Data Refresh to calculate stock coverage and replenishment recommendations.")
    else:
        st.markdown('<div class="scp-section-label">⚡ Filter Replenishment Queue</div>', unsafe_allow_html=True)
        rc1, rc2 = st.columns(2)
        rstatus = rc1.selectbox("Status", ["All"] + sorted(replenishment["status"].dropna().unique().tolist()))
        rep = replenishment.copy()
        if "part_desc" not in rep.columns or rep["part_desc"].isna().all():
            desc_fallback = demand_lines.drop_duplicates("part_code")[["part_code", "part_desc"]]
            rep = rep.drop(columns=["part_desc"], errors="ignore").merge(desc_fallback, on="part_code", how="left")
        rep_parts_avail = rep.drop_duplicates("part_code")[["part_code", "part_desc"]].copy()
        rep_parts_avail["_label"] = rep_parts_avail["part_code"] + " - " + rep_parts_avail["part_desc"].fillna("").str.slice(0, 50)
        with rc2:
            rsel_labels = st.multiselect("Part(s)", rep_parts_avail["_label"].tolist(), label_visibility="visible",
                                           placeholder="Search part code or description...")
        if rstatus != "All":
            rep = rep[rep["status"] == rstatus]
        if rsel_labels:
            sel_parts = rep_parts_avail[rep_parts_avail["_label"].isin(rsel_labels)]["part_code"].tolist()
            rep = rep[rep["part_code"].isin(sel_parts)]
        cols = [c for c in ["part_code", "part_desc", "status", "current_stock", "effective_available_position",
                             "months_of_coverage", "lead_time_days", "criticality", "supply_risk", "recommended_qty", "reason"]
                 if c in rep.columns]
        st.dataframe(rep[cols].rename(columns={"part_desc": "Description"}), use_container_width=True, hide_index=True, height=520)
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# TAB 7: ITEM MASTER & PART TRANSFERS
# ===========================================================================
with tab_item:
    st.markdown('<div class="scp-panel">', unsafe_allow_html=True)
    if len(part_master) > 0:
        part_transfer_integration.render_part_master_section(st, part_master)
    elif len(item_master) == 0:
        st.info("No Item Master data yet &mdash; upload an Item Master file in Data Refresh.")
    else:
        st.markdown('<div class="scp-section-label">🔍 Search / Select Multiple Parts</div>', unsafe_allow_html=True)
        im = item_master.copy()
        im["_label"] = im["part_code"] + " - " + im["part_desc"].fillna("").str.slice(0, 60)
        selected_labels = st.multiselect("Parts", im["_label"].tolist(), label_visibility="collapsed",
                                           placeholder="Type to search part code or description...")
        if selected_labels:
            im_show = im[im["_label"].isin(selected_labels)]
        else:
            search_im = st.text_input("Or filter the full list", placeholder="Search part code or description...")
            im_show = im
            if search_im:
                mask = im["part_code"].str.contains(search_im, case=False, na=False) | im["part_desc"].fillna("").str.contains(search_im, case=False, na=False)
                im_show = im[mask]
        display_cols = [c for c in ["part_code", "part_desc", "item_group", "item_type", "current_stock",
                                      "customer_committed_qty", "open_vendor_qty", "computed_available_qty",
                                      "lead_time_days", "moq", "purchase_multiple", "vendor", "standard_cost",
                                      "part_status_raw", "obsolete_by_oem"] if c in im_show.columns]
        st.dataframe(im_show[display_cols], use_container_width=True, hide_index=True, height=520)
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# TAB 8: AI COPILOT INTELLIGENCE MODULE
# ===========================================================================
with tab_copilot:
    copilot.render_copilot_tab(DB_PATH, project_filter=project_filter)

# ===========================================================================
# TAB 9: DATA REFRESH & INGESTION
# ===========================================================================
with tab_refresh:
    refresh_panel.render_refresh_panel(key_prefix="refresh_tab")
