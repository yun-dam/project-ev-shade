"""Gemini (Vertex AI) classifier: send a primary+context pair, get structured JSON."""
from __future__ import annotations

import json
import time
from pathlib import Path

from google import genai
from google.genai import types

from . import config
from .prompt import PROMPT_VERSION, RESPONSE_SCHEMA, SYSTEM_PROMPT, USER_INSTRUCTION

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True, project=config.PROJECT, location=config.LOCATION
        )
    return _client


def _part(path: Path) -> types.Part:
    return types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png")


def classify_pair(
    primary_path: Path,
    context_path: Path,
    model: str | None = None,
    max_retries: int = config.MAX_RETRIES,
) -> dict:
    """Classify one image pair. Returns dict with classification/confidence/evidence/model."""
    model = model or config.MODEL
    client = get_client()
    contents = [
        USER_INSTRUCTION,
        "IMAGE 1 (PRIMARY, 48m, charger centered):",
        _part(primary_path),
        "IMAGE 2 (CONTEXT, 96m, charger centered):",
        _part(context_path),
    ]
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
            data["error"] = None
            return data
        except Exception as e:  # noqa: BLE001 - retry on any transient/API error
            last_err = e
            time.sleep(min(2**attempt, 16))

    return {
        "classification": "error",
        "confidence": 0.0,
        "evidence": "",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "error": str(last_err)[:300],
    }


if __name__ == "__main__":
    # Smoke test on the known solar-PV site.
    r = classify_pair(
        config.IMAGES_PRIMARY / "169467_primary_b7bb64803110.png",
        config.IMAGES_CONTEXT / "169467_context_b7bb64803110.png",
    )
    print(json.dumps(r, indent=2))
