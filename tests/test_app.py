import io
import os
import tempfile
import importlib

import pytest
from fastapi.testclient import TestClient
from PIL import Image


class DummyProbs:
    def __init__(self):
        self._indices = [0, 1, 2]
        self._values = [0.7, 0.2, 0.1]

    @property
    def top5(self):
        return self._indices

    @property
    def top5conf(self):
        return self._values

    @property
    def top1(self):
        return self._indices[0]

    @property
    def top1conf(self):
        return self._values[0]


class DummyResult:
    def __init__(self):
        self.probs = DummyProbs()


class DummyModel:
    def __init__(self):
        self.names = {0: "Acne", 1: "Eczema", 2: "Melanoma"}

    def __call__(self, path, verbose=False):
        return [DummyResult()]


@pytest.fixture(autouse=True)
def _inject_dummy_model(monkeypatch):
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("SKIP_MODEL_LOAD", "1")
    import app.main as app_module
    importlib.reload(app_module)
    app_module._model = DummyModel()
    yield app_module
    try:
        os.remove(db_path)
    except Exception:
        pass


def _make_image_bytes():
    img = Image.new("RGB", (32, 32), color=(200, 120, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_index_ok(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    res = client.get("/", allow_redirects=False)
    assert res.status_code == 303
    assert res.headers["location"] == "/login"


def test_health_ok(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_signup_login_and_predict_ok(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    res = client.post(
        "/signup",
        data={
            "first_name": "Alex",
            "last_name": "Doe",
            "phone": "1234567890",
            "email": "user@example.com",
            "password": "pass1234",
            "confirm_password": "pass1234",
        },
        allow_redirects=False,
    )
    assert res.status_code == 303
    assert "access_token" in res.cookies

    img_bytes = _make_image_bytes()
    files = {"image": ("test.png", img_bytes, "image/png")}
    res = client.post("/predict", files=files, data={"top_k": "2"})
    assert res.status_code == 200
    assert "Prediction Results" in res.text
    assert "Acne" in res.text


def test_predict_requires_auth(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    files = {"image": ("test.txt", b"not an image", "text/plain")}
    res = client.post("/predict", files=files, data={"top_k": "2"})
    assert res.status_code == 303
    assert res.headers["location"] == "/login"


def test_predict_rejects_empty_file(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    client.post(
        "/signup",
        data={
            "first_name": "Sam",
            "last_name": "Lee",
            "phone": "1234567890",
            "email": "user2@example.com",
            "password": "pass1234",
            "confirm_password": "pass1234",
        },
        allow_redirects=False,
    )
    files = {"image": ("empty.png", b"", "image/png")}
    res = client.post("/predict", files=files, data={"top_k": "2"})
    assert res.status_code == 500
    assert "Empty file" in res.text


def test_admin_page_requires_admin(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    client.post(
        "/signup",
        data={
            "first_name": "Pat",
            "last_name": "Singh",
            "phone": "1234567890",
            "email": "user3@example.com",
            "password": "pass1234",
            "confirm_password": "pass1234",
        },
        allow_redirects=False,
    )
    res = client.get("/admin", allow_redirects=False)
    assert res.status_code == 303


def test_admin_api_requires_admin(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    client.post(
        "/signup",
        data={
            "first_name": "Admin",
            "last_name": "User",
            "phone": "1234567890",
            "email": "admin@example.com",
            "password": "pass1234",
            "confirm_password": "pass1234",
        },
        allow_redirects=False,
    )
    # promote to admin directly in DB
    from app import db as db_module
    from app import models as models_module

    SessionLocal = db_module.get_session_local()
    db = SessionLocal()
    try:
        user = db.query(models_module.User).filter_by(email="admin@example.com").first()
        user.role = "admin"
        db.commit()
    finally:
        db.close()

    res = client.get("/api/admin/users")
    assert res.status_code == 200
    assert "users" in res.json()


def test_admin_can_change_role(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    client.post(
        "/signup",
        data={
            "first_name": "Admin",
            "last_name": "User",
            "phone": "1234567890",
            "email": "admin2@example.com",
            "password": "pass1234",
            "confirm_password": "pass1234",
        },
        allow_redirects=False,
    )
    client.post(
        "/signup",
        data={
            "first_name": "Normal",
            "last_name": "User",
            "phone": "1234567890",
            "email": "user4@example.com",
            "password": "pass1234",
            "confirm_password": "pass1234",
        },
        allow_redirects=False,
    )
    from app import db as db_module
    from app import models as models_module

    SessionLocal = db_module.get_session_local()
    db = SessionLocal()
    try:
        admin = db.query(models_module.User).filter_by(email="admin2@example.com").first()
        admin.role = "admin"
        target = db.query(models_module.User).filter_by(email="user4@example.com").first()
        target_id = target.id
        db.commit()
    finally:
        db.close()

    res = client.post(f"/api/admin/users/{target_id}/role", data={"role": "admin"})
    assert res.status_code == 200
    assert res.json()["role"] == "admin"


def test_admin_can_delete_user(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    client.post(
        "/signup",
        data={
            "first_name": "Admin",
            "last_name": "User",
            "phone": "1234567890",
            "email": "admin3@example.com",
            "password": "pass1234",
            "confirm_password": "pass1234",
        },
        allow_redirects=False,
    )
    client.post(
        "/signup",
        data={
            "first_name": "Target",
            "last_name": "User",
            "phone": "1234567890",
            "email": "target@example.com",
            "password": "pass1234",
            "confirm_password": "pass1234",
        },
        allow_redirects=False,
    )
    from app import db as db_module
    from app import models as models_module

    SessionLocal = db_module.get_session_local()
    db = SessionLocal()
    try:
        admin = db.query(models_module.User).filter_by(email="admin3@example.com").first()
        admin.role = "admin"
        target = db.query(models_module.User).filter_by(email="target@example.com").first()
        target_id = target.id
        db.commit()
    finally:
        db.close()

    res = client.post(f"/api/admin/users/{target_id}", data={"method": "delete"})
    assert res.status_code == 200
    assert res.json()["status"] == "deleted"
