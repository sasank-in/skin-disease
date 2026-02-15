import os
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from sqlalchemy.orm import Session

APP_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_ROOT / "templates"))

MODEL_PATH = Path(os.getenv("MODEL_PATH", "checkpoints/best.pt"))

app = FastAPI(title="Skin Disease Classifier", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(APP_ROOT / "static")), name="static")

_model = None

from .db import get_db, init_db
from .models import Feedback
from .assistant_ai import generate_assistant_insight
from .recommendations import fetch_practo_clinics, get_city_specialists, get_hospitals


@app.on_event("startup")
def _load_model() -> None:
    global _model
    init_db()
    if os.getenv("SKIP_MODEL_LOAD") == "1":
        return
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Set MODEL_PATH or place best.pt in checkpoints/."
        )
    _model = YOLO(str(MODEL_PATH))


@app.get("/", response_class=HTMLResponse)
def index():
    return RedirectResponse(url="/remedy", status_code=303)


@app.get("/remedy", response_class=HTMLResponse)
def remedy_page(request: Request):
    return TEMPLATES.TemplateResponse(
        "remedy.html",
        {
            "request": request,
            "result": None,
            "error": None,
            "hospitals": None,
            "primary_disease": None,
            "feedback_saved": None,
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    image: UploadFile = File(...),
    top_k: int = Form(3),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        return TEMPLATES.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "error": "Please upload a valid image file.",
            },
            status_code=400,
        )

    tmp_path = None
    filename = image.filename or ""
    try:
        contents = await image.read()
        if not contents:
            raise ValueError("Empty file.")

        suffix = Path(image.filename).suffix if image.filename else ""
        if not suffix and image.content_type:
            if image.content_type == "image/jpeg":
                suffix = ".jpg"
            elif image.content_type == "image/png":
                suffix = ".png"
            elif image.content_type == "image/webp":
                suffix = ".webp"
        if not suffix:
            suffix = ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        if _model is None:
            raise RuntimeError("Model not loaded.")

        results = _model(tmp_path, verbose=False)[0]
        probs = results.probs
        names = _model.names

        top_k = max(1, min(int(top_k), len(names)))

        if top_k <= 5:
            top_indices = list(probs.top5)[:top_k]
            top_scores = list(probs.top5conf)[:top_k]
        else:
            # Fallback for k > 5 using raw probability tensor
            arr = probs.data
            try:
                import numpy as np

                arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.array(arr)
                top_indices = np.argsort(arr)[::-1][:top_k].tolist()
                top_scores = arr[top_indices].tolist()
            except Exception:
                # Safe fallback to top5 if numpy/torch is unavailable
                top_indices = list(probs.top5)[:5]
                top_scores = list(probs.top5conf)[:5]

        predictions = []
        for idx, score in zip(top_indices, top_scores):
            predictions.append(
                {
                    "label": names[int(idx)],
                    "confidence": float(score),
                }
            )
    except Exception as exc:
        return TEMPLATES.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "error": f"Prediction failed: {exc}",
            },
            status_code=500,
        )
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    primary_disease = predictions[0]["label"] if predictions else None
    hospitals = get_hospitals(primary_disease) if primary_disease else None

    return TEMPLATES.TemplateResponse(
        "remedy.html",
        {
            "request": request,
            "result": predictions,
            "error": None,
            "hospitals": hospitals,
            "primary_disease": primary_disease,
            "feedback_saved": None,
            "upload_name": filename,
        },
    )


@app.post("/feedback", response_class=HTMLResponse)
def submit_feedback(
    request: Request,
    disease: str = Form(...),
    rating: int = Form(...),
    comments: str = Form(""),
    email: str = Form(""),
    db: Session = Depends(get_db),
):
    rating = max(1, min(int(rating), 5))
    feedback = Feedback(
        disease=disease,
        rating=rating,
        comments=comments.strip() or None,
        email=email.strip() or None,
    )
    db.add(feedback)
    db.commit()
    return TEMPLATES.TemplateResponse(
        "remedy.html",
        {
            "request": request,
            "result": None,
            "error": None,
            "hospitals": None,
            "feedback_saved": True,
            "primary_disease": None,
        },
    )


@app.get("/assistant", response_class=HTMLResponse)
def assistant_page(request: Request):
    return TEMPLATES.TemplateResponse(
        "assistant.html",
        {
            "request": request,
            "assistant_result": None,
        },
    )


@app.post("/assistant", response_class=HTMLResponse)
def assistant(
    request: Request,
    symptoms: str = Form(...),
    duration: str = Form(""),
):
    assistant_result, assistant_error = generate_assistant_insight(symptoms, duration)

    return TEMPLATES.TemplateResponse(
        "assistant.html",
        {
            "request": request,
            "assistant_result": assistant_result,
            "assistant_error": assistant_error,
        },
    )


@app.get("/specialist", response_class=HTMLResponse)
def specialist_page(request: Request, disease: str = "", location: str = ""):
    disease = disease.strip()
    location = location.strip()
    hospitals = get_hospitals(disease) if disease else None
    clinics, clinic_error = fetch_practo_clinics(location)
    city_specialists = get_city_specialists(location)
    return TEMPLATES.TemplateResponse(
        "specialist.html",
        {
            "request": request,
            "disease": disease,
            "hospitals": hospitals,
            "location": location,
            "clinics": clinics,
            "clinic_error": clinic_error,
            "city_specialists": city_specialists,
        },
    )


@app.post("/specialist", response_class=HTMLResponse)
def specialist_submit(request: Request, disease: str = Form(""), location: str = Form("")):
    disease = disease.strip()
    location = location.strip()
    hospitals = get_hospitals(disease) if disease else None
    clinics, clinic_error = fetch_practo_clinics(location)
    city_specialists = get_city_specialists(location)
    return TEMPLATES.TemplateResponse(
        "specialist.html",
        {
            "request": request,
            "disease": disease,
            "hospitals": hospitals,
            "location": location,
            "clinics": clinics,
            "clinic_error": clinic_error,
            "city_specialists": city_specialists,
        },
    )
