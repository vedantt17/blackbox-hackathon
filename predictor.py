"""
predictor.py  —  Approach A: LLM Hybrid Predictor

Takes a persona text + new question + answer options,
calls the Claude API, and returns a structured prediction.

Swap PROVIDER to "groq" or "openai" if you don't have Anthropic credits.
"""

import os
import json
import requests
from typing import Optional

# ── Config ───────────────────────────────────────────────────────────────────
PROVIDER = "gemini"            # "anthropic" | "groq" | "openai" | "gemini"
ANTHROPIC_MODEL = "claude-haiku-4-5"
GROQ_MODEL      = "llama-3.1-8b-instant"
OPENAI_MODEL    = "gpt-4o-mini"
GEMINI_MODEL    = "gemini-2.5-flash"   # free tier, current model as of 2026

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")


SYSTEM_PROMPT = """You are a behavioral prediction engine. You will be given a detailed psychological 
profile of a real person, followed by a survey question and its answer options.

Your task: predict exactly how THIS specific person would answer the question, based on their 
personality traits, biases, values, and behavioral patterns. 

Do NOT give the average human answer. Do NOT give the "correct" answer. 
Give the answer this specific person would most likely choose, given who they are.

Respond ONLY with valid JSON in this exact format:
{
  "predicted_answer": <the exact answer text or numeric value>,
  "predicted_option_number": <integer option number, 1-indexed, or null if open-ended>,
  "confidence": <float 0.0-1.0>,
  "reasoning": <1-2 sentences explaining why this person would answer this way>
}"""


def build_prediction_prompt(
    persona_text: str,
    question: str,
    options: Optional[list] = None,
    distribution_hint: Optional[str] = None,
) -> str:
    """
    Construct the user message for the LLM.
    
    Args:
        persona_text: Full text persona for this person
        question: The new survey question text
        options: List of answer options (if multiple choice). None if open-ended/numeric.
        distribution_hint: Optional note about population-level distribution for this question.
    """
    msg = f"{persona_text}\n\n"
    msg += "━" * 50 + "\n"
    msg += f"NEW QUESTION TO PREDICT:\n{question}\n\n"

    if options:
        msg += "ANSWER OPTIONS:\n"
        for i, opt in enumerate(options, 1):
            msg += f"  {i}. {opt}\n"
        if distribution_hint:
            msg += f"\nPOPULATION CONTEXT: {distribution_hint}\n"
        msg += "\nPredict which option THIS SPECIFIC PERSON would choose based on their unique profile."
    else:
        msg += "This is an open-ended or numeric question. Predict their specific answer value."

    return msg


def call_anthropic(prompt: str) -> dict:
    """Call Anthropic API and parse response."""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 300,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    content = resp.json()["content"][0]["text"]
    return json.loads(content)


def call_groq(prompt: str) -> dict:
    """Call Groq API (free tier) and parse response."""
    import time, re
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.1,
    }

    for attempt in range(4):
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=body, timeout=30
        )

        # Rate limit — wait and retry
        if resp.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"\n  [rate limit] waiting {wait}s...", end="", flush=True)
            time.sleep(wait)
            continue

        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON object from anywhere in the text
            match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
            # Last resort: parse option number from text
            num_match = re.search(r"\b([1-5])\b", raw)
            if num_match:
                n = int(num_match.group(1))
                return {"predicted_answer": None, "predicted_option_number": n}
            raise

    raise Exception("Groq API failed after 4 retries")


def call_openai(prompt: str) -> dict:
    """Call OpenAI API and parse response."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)


# Population distribution hints for common question patterns
# These anchor the LLM to realistic base rates and prevent defaulting to middle options
DISTRIBUTION_HINTS = {
    "social_ability":   "Most people (>70%) report being able to adapt their behavior in social situations. Only predict low ability if the persona shows strong evidence of social rigidity.",
    "phone_morning":    "Most people check their phone within 5 minutes of waking. Only predict disagreement if the persona shows strong evidence of low tech engagement or deliberate boundaries.",
    "conscientiousness": "Distribution is roughly even across options. Base your prediction entirely on the persona's conscientiousness and self-discipline signals.",
    "uniqueness":       "Most people lean toward conformity (Disagree). Only predict strong uniqueness-seeking if the persona shows clear nonconformist signals.",
    "default":          "Base your prediction on this specific person's profile. Do not default to neutral or middle options without evidence from the persona.",
}

def _get_distribution_hint(question: str, options: Optional[list]) -> str:
    """Infer the most relevant distribution hint for a given question."""
    q_lower = question.lower()
    if any(w in q_lower for w in ["alter my behavior", "adjust my behavior", "social situation", "fit in"]):
        return DISTRIBUTION_HINTS["social_ability"]
    if any(w in q_lower for w in ["phone", "check", "waking", "morning"]):
        return DISTRIBUTION_HINTS["phone_morning"]
    if any(w in q_lower for w in ["lazy", "disorganized", "thorough", "reliable", "efficient"]):
        return DISTRIBUTION_HINTS["conscientiousness"]
    if any(w in q_lower for w in ["stand out", "unique", "different", "original", "duplicate", "popular"]):
        return DISTRIBUTION_HINTS["uniqueness"]
    return DISTRIBUTION_HINTS["default"]


def call_gemini(prompt: str) -> dict:
    """Call Google Gemini API (free tier: 15 req/min, 1500/day)."""
    import time, re
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "contents": [
            {
                "parts": [
                    {"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1024,
        }
    }

    for attempt in range(4):
        resp = requests.post(url, json=body, headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}, timeout=30)

        if resp.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"\n  [rate limit] waiting {wait}s...", end="", flush=True)
            time.sleep(wait)
            continue

        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
            num_match = re.search(r"\b([1-6])\b", raw)
            if num_match:
                return {"predicted_answer": None, "predicted_option_number": int(num_match.group(1))}
            raise

    raise Exception("Gemini API failed after 4 retries")


def predict_single(
    person_id: str,
    persona_text: str,
    question: str,
    options: Optional[list] = None,
) -> dict:
    """
    Predict one person's answer to one question.
    
    Returns:
        {
            "person_id": str,
            "predicted_answer": str or float,
            "predicted_option_number": int or None,
            "confidence": float,
            "reasoning": str,
            "provider": str,
        }
    """
    hint = _get_distribution_hint(question, options) if options else None
    prompt = build_prediction_prompt(persona_text, question, options, distribution_hint=hint)

    try:
        if PROVIDER == "anthropic":
            result = call_anthropic(prompt)
        elif PROVIDER == "groq":
            result = call_groq(prompt)
        elif PROVIDER == "openai":
            result = call_openai(prompt)
        elif PROVIDER == "gemini":
            result = call_gemini(prompt)
        else:
            raise ValueError(f"Unknown provider: {PROVIDER}")

        result["person_id"] = person_id
        result["provider"] = PROVIDER
        return result

    except json.JSONDecodeError as e:
        return {
            "person_id": person_id,
            "predicted_answer": None,
            "predicted_option_number": None,
            "confidence": 0.0,
            "reasoning": f"JSON parse error: {e}",
            "provider": PROVIDER,
            "error": True,
        }
    except Exception as e:
        return {
            "person_id": person_id,
            "predicted_answer": None,
            "predicted_option_number": None,
            "confidence": 0.0,
            "reasoning": f"API error: {e}",
            "provider": PROVIDER,
            "error": True,
        }


def predict_all(
    personas: dict,
    question: str,
    options: Optional[list] = None,
) -> list[dict]:
    """
    Predict all 233 people's answers to one question.
    
    Args:
        personas: {person_id -> persona_text}
        question: The question text
        options: Answer options list (or None for open-ended)
    
    Returns:
        List of prediction dicts
    """
    import time
    results = []
    total = len(personas)
    for i, (person_id, persona_text) in enumerate(personas.items(), 1):
        print(f"  [{i}/{total}] Predicting {person_id}...", end="\r")
        result = predict_single(person_id, persona_text, question, options)
        results.append(result)
        # Rate limit management per provider
        # Gemini 2.5 flash free tier: ~10 req/min, pause 6s every call
        # Groq: 30 req/min -> pause every 25 calls
        if PROVIDER == "gemini":
            time.sleep(6)  # 6s between every call = 10 req/min, safely under limit
        elif PROVIDER == "groq" and i % 25 == 0:
            time.sleep(2)
    print(f"\n[predictor] Done — {len(results)} predictions")
    return results


# ── quick test (no API key needed, just checks structure) ────────────────────
if __name__ == "__main__":
    sample_persona = "=== DIGITAL TWIN: test123 ===\nHigh Conscientiousness, Low Neuroticism..."
    sample_question = "How much do you agree: 'I prefer to plan ahead rather than be spontaneous'?"
    sample_options = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]

    prompt = build_prediction_prompt(sample_persona, sample_question, sample_options)
    print("=== SAMPLE PROMPT ===")
    print(prompt)
    print("\nTo test with real API, set ANTHROPIC_API_KEY env var and run predict_single()")