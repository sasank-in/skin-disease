import json
import os
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


def generate_assistant_insight(symptoms: str, duration: str) -> Tuple[Dict[str, str], str | None]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return _fallback_insight(symptoms, duration), "GEMINI_API_KEY is not set."

    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    system = (
        "You are a clinical support assistant for skin concerns. "
        "You do not diagnose; you provide possible explanations, seriousness level, "
        "red flags, self-care, and next steps. Keep it concise and safe. "
        "Return STRICT JSON with keys: summary, seriousness, next_steps, red_flags, self_care, follow_up_questions."
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

    try:
        resp = requests.post(
            endpoint,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
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
        if not text:
            return _fallback_insight(symptoms, duration), "Gemini returned empty response."

        # Attempt JSON extraction if model returns structured output
        try:
            maybe_json = text[text.find("{") : text.rfind("}") + 1]
            obj = json.loads(maybe_json) if maybe_json else None
            if isinstance(obj, dict):
                return {
                    "summary": obj.get("summary", ""),
                    "seriousness": obj.get("seriousness", ""),
                    "next_steps": obj.get("next_steps", ""),
                    "red_flags": obj.get("red_flags", ""),
                    "self_care": obj.get("self_care", ""),
                    "follow_up_questions": obj.get("follow_up_questions", []),
                }, None
        except Exception:
            pass

        # Fallback: map plain text into summary
        return {
            "summary": text.strip(),
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
    except Exception as exc:
        return _fallback_insight(symptoms, duration), f"Gemini request failed: {exc}"
