import json
import os
import re
from typing import Dict, Tuple

import requests


def _fallback_insight(symptoms: str, duration: str) -> Dict[str, str]:
    summary = "We could not interpret the input. Please add more detail."
    seriousness = "Unclear"
    next_steps = "Provide symptom location, onset, and any triggers."
    red_flags = "Severe pain, rapid spreading, bleeding, or fever."
    self_care = "Keep the area clean and avoid known irritants."

    text = f"{symptoms} {duration}".lower()
    if any(word in text for word in ["bleeding", "black", "rapidly growing", "irregular", "ulcer"]):
        seriousness = "High"
        summary = "Symptoms suggest a potentially serious skin concern."
        next_steps = "Seek dermatologist evaluation soon."
    elif any(word in text for word in ["itch", "rash", "dry", "redness", "flaking"]):
        seriousness = "Moderate"
        summary = "Symptoms align with inflammatory or allergic skin conditions."
        next_steps = "Consider gentle skincare and consult a clinician if persistent."
    elif len(symptoms.strip()) > 0:
        seriousness = "Low to moderate"
        summary = "Symptoms appear mild, but monitor for changes."
        next_steps = "If worsening or persistent, consult a specialist."

    return {
        "summary": summary,
        "seriousness": seriousness,
        "next_steps": next_steps,
        "red_flags": red_flags,
        "self_care": self_care,
    }


def _parse_model_text(text: str) -> Tuple[Dict[str, str], str | None]:
    if not text:
        return {}, "Empty response."
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    def _coerce(obj: dict) -> Dict[str, str]:
        red_flags = obj.get("red_flags", "")
        if isinstance(red_flags, list):
            red_flags = "; ".join([str(x) for x in red_flags if x])
        follow_up = obj.get("follow_up_questions", [])
        if isinstance(follow_up, str):
            follow_up = [follow_up]
        return {
            "summary": obj.get("summary", ""),
            "seriousness": obj.get("seriousness", ""),
            "next_steps": obj.get("next_steps", ""),
            "red_flags": red_flags,
            "self_care": obj.get("self_care", ""),
            "follow_up_questions": follow_up,
        }

    def _repair_json(raw: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", raw)

    def _escape_newlines_in_strings(raw: str) -> str:
        out = []
        in_str = False
        escape = False
        for ch in raw:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == "\"":
                in_str = not in_str
                out.append(ch)
                continue
            if in_str and ch in ("\n", "\r"):
                out.append("\\n")
                continue
            out.append(ch)
        return "".join(out)

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return _coerce(obj), None
    except Exception:
        pass

    try:
        maybe_json = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
        maybe_json = _escape_newlines_in_strings(_repair_json(maybe_json))
        obj = json.loads(maybe_json) if maybe_json else None
        if isinstance(obj, dict):
            return _coerce(obj), None
    except Exception:
        pass

    return {
        "summary": cleaned,
        "seriousness": "See summary",
        "next_steps": "Follow the guidance above.",
        "red_flags": "Seek urgent care if severe pain, fever, bleeding, or rapid change.",
        "self_care": "Avoid irritants and keep the area clean.",
        "follow_up_questions": [
            "Where exactly is the affected area?",
            "When did it start and is it changing?",
            "Any known triggers or new products?",
        ],
    }, None


def _call_groq(symptoms: str, duration: str, api_key: str) -> Tuple[Dict[str, str], str | None]:
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    base_url = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
    endpoint = f"{base_url}/chat/completions"

    system = (
        "You are a clinical support assistant for skin concerns. "
        "You do not diagnose; you provide possible explanations, seriousness level, "
        "red flags, self-care, and next steps. Keep it concise and safe. "
        "Return STRICT JSON with keys: summary, seriousness, next_steps, red_flags, self_care, follow_up_questions. "
        "Use double quotes, no trailing commas, and use \\n for line breaks inside strings."
    )
    user = f"Symptoms: {symptoms}\nDuration: {duration}".strip()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.4,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
    }

    def _post(body: dict):
        return requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=30,
        )

    try:
        resp = _post(payload)
        if resp.status_code == 400 and "response_format" in payload:
            payload.pop("response_format", None)
            resp = _post(payload)
        if resp.status_code == 429:
            return (
                _fallback_insight(symptoms, duration),
                "Groq rate limit or quota exceeded. Check billing or try later.",
            )
        if resp.status_code != 200:
            return _fallback_insight(symptoms, duration), f"Groq error: {resp.status_code}"
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        result, parse_err = _parse_model_text(text)
        if parse_err:
            return _fallback_insight(symptoms, duration), f"Groq returned empty response."
        return result, None
    except Exception as exc:
        return _fallback_insight(symptoms, duration), f"Groq request failed: {exc}"


def _call_gemini(symptoms: str, duration: str, api_key: str) -> Tuple[Dict[str, str], str | None]:
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    if model.startswith("models/"):
        model = model.split("/", 1)[1]
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    system = (
        "You are a clinical support assistant for skin concerns. "
        "You do not diagnose; you provide possible explanations, seriousness level, "
        "red flags, self-care, and next steps. Keep it concise and safe. "
        "Return STRICT JSON with keys: summary, seriousness, next_steps, red_flags, self_care, follow_up_questions. "
        "Use double quotes, no trailing commas, and use \\n for line breaks inside strings."
    )
    user = f"Symptoms: {symptoms}\nDuration: {duration}".strip()

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": system}]},
            {"role": "user", "parts": [{"text": user}]},
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 500,
        },
    }

    def _list_models_error() -> str:
        try:
            list_url = "https://generativelanguage.googleapis.com/v1beta/models"
            resp = requests.get(
                list_url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                timeout=30,
            )
            if resp.status_code != 200:
                return "Model not found. Unable to list available models."
            models = resp.json().get("models", [])
            names = [m.get("name") for m in models if m.get("name")]
            preview = ", ".join(names[:8]) if names else "No models returned."
            return f"Model not found. Available models (partial): {preview}"
        except Exception:
            return "Model not found. Unable to list available models."

    try:
        resp = requests.post(
            endpoint,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if resp.status_code == 404:
            return _fallback_insight(symptoms, duration), _list_models_error()
        if resp.status_code == 429:
            return (
                _fallback_insight(symptoms, duration),
                "Gemini rate limit or quota exceeded. Check billing or try later.",
            )
        if resp.status_code != 200:
            return _fallback_insight(symptoms, duration), f"Gemini error: {resp.status_code}"
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        result, parse_err = _parse_model_text(text)
        if parse_err:
            return _fallback_insight(symptoms, duration), "Gemini returned empty response."
        return result, None
    except Exception as exc:
        return _fallback_insight(symptoms, duration), f"Gemini request failed: {exc}"


def generate_assistant_insight(symptoms: str, duration: str) -> Tuple[Dict[str, str], str | None]:
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return _call_groq(symptoms, duration, groq_key)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return _fallback_insight(symptoms, duration), "GROQ_API_KEY or GEMINI_API_KEY is not set."

    return _call_gemini(symptoms, duration, api_key)
