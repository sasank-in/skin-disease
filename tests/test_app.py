import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main as app_module


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
    monkeypatch.setenv("SKIP_MODEL_LOAD", "1")
    app_module._model = DummyModel()
    yield


def _make_image_bytes():
    img = Image.new("RGB", (32, 32), color=(200, 120, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_index_ok():
    client = TestClient(app_module.app)
    res = client.get("/")
    assert res.status_code == 200
    assert "SkinDx Insight" in res.text


def test_health_ok():
    client = TestClient(app_module.app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_predict_ok():
    client = TestClient(app_module.app)
    img_bytes = _make_image_bytes()
    files = {"image": ("test.png", img_bytes, "image/png")}
    res = client.post("/predict", files=files, data={"top_k": "2"})
    assert res.status_code == 200
    assert "Prediction Results" in res.text
    assert "Acne" in res.text


def test_predict_rejects_non_image():
    client = TestClient(app_module.app)
    files = {"image": ("test.txt", b"not an image", "text/plain")}
    res = client.post("/predict", files=files, data={"top_k": "2"})
    assert res.status_code == 400
    assert "valid image" in res.text


def test_predict_rejects_empty_file():
    client = TestClient(app_module.app)
    files = {"image": ("empty.png", b"", "image/png")}
    res = client.post("/predict", files=files, data={"top_k": "2"})
    assert res.status_code == 500
    assert "Empty file" in res.text
