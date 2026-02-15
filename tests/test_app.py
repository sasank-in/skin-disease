import io
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
    monkeypatch.setenv("SKIP_MODEL_LOAD", "1")
    import app.main as app_module
    importlib.reload(app_module)
    app_module._model = DummyModel()
    yield app_module


def _make_image_bytes():
    img = Image.new("RGB", (32, 32), color=(200, 120, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_index_ok(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app, follow_redirects=False)
    res = client.get("/")
    assert res.status_code == 303
    assert res.headers["location"] == "/remedy"


def test_health_ok(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_predict_ok(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    img_bytes = _make_image_bytes()
    files = {"image": ("test.png", img_bytes, "image/png")}
    res = client.post("/predict", files=files, data={"top_k": "2"})
    assert res.status_code == 200
    assert "Prediction Results" in res.text
    assert "Acne" in res.text


def test_feedback_saved(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    res = client.post(
        "/feedback",
        data={"disease": "Acne", "rating": "4", "comments": "Helpful", "email": "a@b.com"},
    )
    assert res.status_code == 200
    assert "feedback was saved" in res.text.lower()


def test_predict_rejects_non_image(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    files = {"image": ("test.txt", b"not an image", "text/plain")}
    res = client.post("/predict", files=files, data={"top_k": "2"})
    assert res.status_code == 400
    assert "valid image" in res.text


def test_predict_rejects_empty_file(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    files = {"image": ("empty.png", b"", "image/png")}
    res = client.post("/predict", files=files, data={"top_k": "2"})
    assert res.status_code == 500
    assert "Empty file" in res.text


def test_assistant_page_and_post(_inject_dummy_model):
    client = TestClient(_inject_dummy_model.app)
    res = client.get("/assistant")
    assert res.status_code == 200
    assert "AI Health Assistant" in res.text
    res = client.post("/assistant", data={"symptoms": "itchy rash", "duration": "2 weeks"})
    assert res.status_code == 200
    assert "Assistant Insight" in res.text


def test_specialist_page_and_post(_inject_dummy_model):
    import os
    os.environ["PRACTO_PROVIDER"] = "stub"
    client = TestClient(_inject_dummy_model.app)
    res = client.get("/specialist")
    assert res.status_code == 200
    assert "Find Specialist" in res.text
    res = client.post("/specialist", data={"disease": "Acne", "location": "bangalore"})
    assert res.status_code == 200
    assert "Recommended Facilities" in res.text
