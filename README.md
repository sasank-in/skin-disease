# SkinDx Insight (FastAPI)

FastAPI app to serve your YOLOv8 skin disease classifier from `checkpoints/best.pt`.

## Prerequisites
- Python 3.9+

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Start The App
```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`.

## Health Check
```bash
http://127.0.0.1:8000/health
```

## Model Path
By default the app loads `checkpoints/best.pt`.
Set an explicit path if needed:

```bash
set MODEL_PATH=C:\path\to\best.pt
uvicorn main:app --reload
```

## Run Tests
```bash
pytest -q
```

## Notes
- Tests bypass model load by setting `SKIP_MODEL_LOAD=1` in the test fixture.

## Demo Credentials
To seed demo users on startup, set:
```bash
set SEED_DEMO_USERS=1
set DEMO_ADMIN_EMAIL=admin@demo.local
set DEMO_ADMIN_PASSWORD=Admin@12345
set DEMO_USER_EMAIL=user@demo.local
set DEMO_USER_PASSWORD=User@12345
```

Then restart the app and log in with the credentials above.
