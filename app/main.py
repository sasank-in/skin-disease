import os
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile, HTTPException
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

from .auth import create_access_token, decode_access_token, get_password_hash, verify_password
from .db import get_db, init_db
from .models import User


@app.on_event("startup")
def _load_model() -> None:
    global _model
    init_db()
    _seed_demo_users()
    if os.getenv("SKIP_MODEL_LOAD") == "1":
        return
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Set MODEL_PATH or place best.pt in checkpoints/."
        )
    _model = YOLO(str(MODEL_PATH))


def _seed_demo_users() -> None:
    if os.getenv("SEED_DEMO_USERS") != "1":
        return
    from .db import get_session_local

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        admin_email = os.getenv("DEMO_ADMIN_EMAIL", "admin@demo.local")
        admin_password = os.getenv("DEMO_ADMIN_PASSWORD", "Admin@12345")
        user_email = os.getenv("DEMO_USER_EMAIL", "user@demo.local")
        user_password = os.getenv("DEMO_USER_PASSWORD", "User@12345")

        try:
            admin_hash = get_password_hash(admin_password)
            user_hash = get_password_hash(user_password)
        except ValueError as exc:
            print(f"Demo user seed skipped: {exc}")
            return

        if not db.query(User).filter(User.email == admin_email).first():
            db.add(
                User(
                    first_name="Demo",
                    last_name="Admin",
                    phone="9999999999",
                    email=admin_email,
                    hashed_password=admin_hash,
                    role="admin",
                )
            )
        if not db.query(User).filter(User.email == user_email).first():
            db.add(
                User(
                    first_name="Demo",
                    last_name="User",
                    phone="8888888888",
                    email=user_email,
                    hashed_password=user_hash,
                    role="user",
                )
            )
        db.commit()
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):
    user = _get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return TEMPLATES.TemplateResponse(
        "index.html",
        {"request": request, "result": None, "error": None, "user": user},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/signup", response_class=HTMLResponse)
def signup_get(request: Request):
    return TEMPLATES.TemplateResponse(
        "signup.html",
        {"request": request, "error": None},
    )


@app.post("/signup", response_class=HTMLResponse)
def signup_post(
    request: Request,
    first_name: str = Form(...),
    last_name: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
):
    if password != confirm_password:
        return TEMPLATES.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Passwords do not match."},
            status_code=400,
        )
    if not all([first_name.strip(), last_name.strip(), phone.strip(), email.strip(), password.strip()]):
        return TEMPLATES.TemplateResponse(
            "signup.html",
            {"request": request, "error": "All fields are required."},
            status_code=400,
        )
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return TEMPLATES.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Account already exists."},
            status_code=400,
        )
    try:
        hashed = get_password_hash(password)
    except ValueError as exc:
        return TEMPLATES.TemplateResponse(
            "signup.html",
            {"request": request, "error": str(exc)},
            status_code=400,
        )
    user = User(
        first_name=first_name,
        last_name=last_name,
        phone=phone,
        email=email,
        hashed_password=hashed,
        role="user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(str(user.id))
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie("access_token", token, httponly=True, samesite="lax")
    return response


@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return TEMPLATES.TemplateResponse(
        "login.html",
        {"request": request, "error": None},
    )


@app.post("/login", response_class=HTMLResponse)
def login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == email).first()
    try:
        valid = user and verify_password(password, user.hashed_password)
    except ValueError as exc:
        return TEMPLATES.TemplateResponse(
            "login.html",
            {"request": request, "error": str(exc)},
            status_code=400,
        )
    if not valid:
        return TEMPLATES.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid email or password."},
            status_code=401,
        )
    token = create_access_token(str(user.id))
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie("access_token", token, httponly=True, samesite="lax")
    return response


@app.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("access_token")
    return response


def _get_current_user(request: Request, db: Session):
    token = request.cookies.get("access_token")
    if not token:
        return None
    user_id = decode_access_token(token)
    if not user_id:
        return None
    return db.query(User).filter(User.id == int(user_id)).first()


def _require_admin(request: Request, db: Session):
    if request is None:
        return None
    user = _get_current_user(request, db)
    if not user:
        return None
    if user.role != "admin":
        return None
    return user


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, db: Session = Depends(get_db)):
    user = _require_admin(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    users = db.query(User).all()
    return TEMPLATES.TemplateResponse(
        "admin.html",
        {"request": request, "user": user, "users": users},
    )


@app.get("/api/admin/users")
def admin_users(request: Request, db: Session = Depends(get_db)):
    user = _require_admin(request, db)
    if not user:
        raise HTTPException(status_code=403, detail="forbidden")
    users = db.query(User).all()
    return {
        "users": [
            {
                "id": u.id,
                "first_name": u.first_name,
                "last_name": u.last_name,
                "email": u.email,
                "phone": u.phone,
                "role": u.role,
            }
            for u in users
        ]
    }


@app.post("/api/admin/users/{user_id}/role")
def admin_set_role(user_id: int, role: str = Form(...), request: Request = None, db: Session = Depends(get_db)):
    user = _require_admin(request, db)
    if not user:
        raise HTTPException(status_code=403, detail="forbidden")
    if role not in {"admin", "user"}:
        raise HTTPException(status_code=400, detail="invalid role")
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="user not found")
    target.role = role
    db.commit()
    return {"status": "ok", "id": target.id, "role": target.role}


@app.post("/api/admin/users/{user_id}")
def admin_delete_user(
    user_id: int,
    request: Request = None,
    method: str = Form(None),
    db: Session = Depends(get_db),
):
    user = _require_admin(request, db)
    if not user:
        raise HTTPException(status_code=403, detail="forbidden")
    if method != "delete":
        raise HTTPException(status_code=400, detail="invalid request")
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="user not found")
    if target.id == user.id:
        raise HTTPException(status_code=400, detail="cannot delete self")
    db.delete(target)
    db.commit()
    return {"status": "deleted", "id": user_id}


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    image: UploadFile = File(...),
    top_k: int = Form(3),
    db: Session = Depends(get_db),
):
    user = _get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not image.content_type or not image.content_type.startswith("image/"):
        return TEMPLATES.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "error": "Please upload a valid image file.",
                "user": user,
            },
            status_code=400,
        )

    tmp_path = None
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
                "user": user,
            },
            status_code=500,
        )
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": predictions,
            "error": None,
            "user": user,
        },
    )
