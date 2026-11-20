# SCIIP Insight Hub

SCIIP is a Streamlit application for semiconductor supply-chain and inventory intelligence. It reads `data/sciip.db` and provides dashboards for customers, parts, transactions, forecasts, replenishment, item master data, and refresh operations.

The repository also includes the SCIIP AI Copilot in the **AI Copilot** dashboard tab. It retrieves relevant records from SQLite and uses them as grounded evidence for natural-language answers.

## Requirements

- Python 3.14 or newer
- This repository
- The prepared database at `data/sciip.db`
- Optional provider API keys for the AI Copilot

Use `python -m ...` commands so pip, Streamlit, and pytest use the same Python installation.

## First-time setup

### Windows PowerShell

```powershell
cd "D:\path\to\sciip version\sciip"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, run this once for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### macOS/Linux

```bash
cd /path/to/sciip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Do not commit `.env`. It is ignored by Git. Never put real API keys in `.env.example`.

## Configure the AI Copilot

Edit `.env` and configure one or more OpenAI-compatible providers. Slots are attempted in order from `1` through `4`. The provider selected in Copilot settings is attempted first; configured slots are used as automatic fallbacks afterward.

Example:

```dotenv
SCIIP_AI_PROVIDER_1=gemini
SCIIP_AI_API_KEY_1=your-gemini-key
SCIIP_AI_MODEL_1=gemini-3.8-flash
SCIIP_AI_BASE_URL_1=https://generativelanguage.googleapis.com/v1beta/openai/

SCIIP_AI_PROVIDER_2=groq
SCIIP_AI_API_KEY_2=your-groq-key
SCIIP_AI_MODEL_2=llama-3.1-8b-instant
SCIIP_AI_BASE_URL_2=https://api.groq.com/openai/v1
```

Available provider names include `gemini`, `groq`, `openai`, `openrouter`, `mistral`, `together`, `ollama`, and `offline`.

API keys remain server-side. The Copilot retrieves bounded evidence from SQLite and shows supporting table names. If all providers fail, it returns grounded database evidence instead of inventing an answer.

## Start the dashboard

From the project directory with `.venv` activated:

```powershell
python -m streamlit run dashboard.py
```

Open the URL shown by Streamlit, normally `http://localhost:8501`.

The AI Copilot remains available in the **AI Copilot** tab. The floating home-page launcher is intentionally disabled.

## Refresh the database

The dashboard has a refresh panel for a new Sales Register and optional supporting files.

For command-line runs:

```powershell
python pipeline.py path\to\sales_register.xlsx
```

Optional files are supplied in this order:

```text
sales_register.xlsx item_master.xlsx customer_commitments.xlsx part_transfer.xlsx
```

The pipeline writes the prepared database to `data/sciip.db` and generated files to `outputs/`.

## Run checks

```powershell
python -m pytest -q
python -m py_compile dashboard.py copilot.py copilot_analytics.py
```

## Important files

- `dashboard.py` — Streamlit application entry point
- `copilot.py` — Copilot UI, providers, fallback, and conversation history
- `copilot_analytics.py` — read-only semantic intent routing and SQLite retrieval
- `pipeline.py` — data ingestion and intelligence pipeline
- `data/sciip.db` — prepared application database
- `.env.example` — safe provider configuration template
- `tests/` — automated tests

## Troubleshooting

### The commands use the wrong Python installation

Activate `.venv` and use module commands:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run dashboard.py
python -m pytest -q
```

### Copilot says providers are unavailable

Check the provider name, API key, model, and OpenAI-compatible base URL in `.env`. Safe error categories include authentication, rate limit, timeout, and model unavailable. API keys are not printed.

### Copilot returns database evidence instead of a natural-language answer

The configured providers failed or no API key is configured. The grounded database fallback is expected and uses current SCIIP records.

### Database not found

Confirm that `data/sciip.db` exists relative to `dashboard.py`. If it is missing, run the pipeline with a valid Sales Register workbook or use the dashboard refresh panel.
