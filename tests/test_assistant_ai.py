import os

from app.assistant_ai import generate_assistant_insight


class _Resp:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data or {}
        self.text = text

    def json(self):
        return self._data


def test_generate_assistant_insight_strips_models_prefix(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "models/gemini-2.0-flash")

    seen = {}

    def _fake_post(url, headers=None, json=None, timeout=30):
        seen["url"] = url
        return _Resp(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '{'
                                        '"summary":"ok",'
                                        '"seriousness":"Low",'
                                        '"next_steps":"n/a",'
                                        '"red_flags":"n/a",'
                                        '"self_care":"n/a",'
                                        '"follow_up_questions":[]'
                                        '}'
                                    )
                                }
                            ]
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("app.assistant_ai.requests.post", _fake_post)

    result, err = generate_assistant_insight("itch", "1 day")
    assert err is None
    assert "models/gemini-2.0-flash:generateContent" in seen["url"]
    assert result["summary"] == "ok"


def test_generate_assistant_insight_uses_groq_when_key_present(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")

    seen = {}

    def _fake_post(url, headers=None, json=None, timeout=30):
        seen["url"] = url
        seen["auth"] = headers.get("Authorization") if headers else None
        return _Resp(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "{"
                                "\"summary\":\"ok\","
                                "\"seriousness\":\"Low\","
                                "\"next_steps\":\"n/a\","
                                "\"red_flags\":\"n/a\","
                                "\"self_care\":\"n/a\","
                                "\"follow_up_questions\":[]"
                                "}"
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("app.assistant_ai.requests.post", _fake_post)

    result, err = generate_assistant_insight("itch", "1 day")
    assert err is None
    assert seen["url"].endswith("/chat/completions")
    assert seen["auth"] == "Bearer test-groq"
    assert result["summary"] == "ok"
