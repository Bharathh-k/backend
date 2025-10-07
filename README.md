# Guardrail Backend (Deployable)

New packaging of the FastAPI backend ready for containerisation/deployment while preserving behaviour.

## Project Structure

```
backend/
  app/
    api/
    core/
    services/
    utils/
    data/
    prompts/
    main.py
    __main__.py
  requirements.txt
  README.md
  .env.example
```

* `app/main.py` – FastAPI application identical in behaviour to the original `guardrail-backend/main.py`.
* `app/core` – analysis helpers and signal generation.
* `app/services` – legacy scripts executed in subprocesses (unchanged logic).
* `app/data` / `app/prompts` – CSV assets and prompt templates.

## Environment Variables

Copy `.env.example` to `.env` (or export values) and override as needed:

- `MONGO_URI`
- `GEMINI_API_KEY`
- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`

At runtime the settings loader looks for variables in the process environment first, then falls back to a local `.env` file located at `backend/.env` (or one in the repository root). Missing values will stop the application during startup, which helps avoid deploying with placeholder keys.

## Installation & Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app  # runs uvicorn via __main__
```

Render defaults to Python 3.13, but several pinned packages expect the Python 3.11 series. The included `runtime.txt` pins deployments (Render or otherwise) to Python 3.11.9 so the build succeeds without recompiling heavy dependencies.

Alternatively, invoke `uvicorn app.main:app` with your preferred process manager.

All filesystem paths inside the code use `app.config.settings`, so ensure the CSV files exist under `app/data/` (already included).
