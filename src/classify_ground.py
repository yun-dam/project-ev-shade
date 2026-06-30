"""Gemini (Vertex AI) classifier for GROUND-LEVEL images.

Sends the 1-3 Google photos of one station (place photos or a Street View capture)
to Gemini with the two-stage ground prompt, and returns structured JSON:
charger_present / charger_confidence / classification / confidence / evidence.

Uses the same Vertex ADC + project (hai-gcp-a-model) as classify.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from google import genai
from google.genai import types

from . import config
from .prompt_ground import (
    PROMPT_VERSION,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    USER_INSTRUCTION,
)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True, project=config.PROJECT, location=config.LOCATION
        )
    return _client


def _part(path: Path) -> types.Part:
    return types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg")


def classify_site(
    image_paths: list[Path],
    model: str | None = None,
    max_retries: int = config.MAX_RETRIES,
) -> dict:
    """Classify one station from its 1-3 ground photos. Returns a result dict."""
    model = model or config.MODEL
    client = get_client()

    contents: list = [USER_INSTRUCTION]
    for i, p in enumerate(image_paths, start=1):
        contents.append(f"IMAGE {i} of {len(image_paths)} (same station):")
        contents.append(_part(p))

    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model, contents=contents, config=cfg
            )
            data = json.loads(resp.text)
            data["model"] = model
            data["prompt_version"] = PROMPT_VERSION
            data["n_images"] = len(image_paths)
            data["error"] = None
            return data
        except Exception as e:  # noqa: BLE001 - retry on any transient/API error
            last_err = e
            time.sleep(min(2**attempt, 16))

    return {
        "charger_present": None,
        "charger_confidence": 0.0,
        "classification": "error",
        "confidence": 0.0,
        "evidence": "",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "n_images": len(image_paths),
        "error": str(last_err)[:300],
    }
