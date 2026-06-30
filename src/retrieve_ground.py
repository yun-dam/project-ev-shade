"""Dual-source ground-level image retrieval for EV charging sites.

For a target ``(lat, lon)`` this discovers the nearby EV charger via the Google
Places API (New) Nearby Search, then fetches the best available real-world image:

  1. If the charger has user-contributed photos -> Place Photos (Photo Media) API.
  2. Otherwise -> Street View Static API, gated by the *free* Street View Metadata
     check so we never pay for a "no imagery available" placeholder canvas.

This complements the top-down NAIP classifier with ground-level / crowdsourced
imagery for the same sites. See NEXT_PLAN.md for the full design rationale.

Auth: this path uses a Google **Maps Platform API key** (``GOOGLE_MAPS_API_KEY``),
which is SEPARATE from the Vertex AI ADC the classifier uses. Enable on the key:
Places API (New) and Street View Static API.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from . import config

# --- Endpoints --------------------------------------------------------------
PLACES_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
PLACES_DETAILS_URL = "https://places.googleapis.com/v1/places/{place_id}"
PLACES_PHOTO_URL = "https://places.googleapis.com/v1/{name}/media"
STREETVIEW_URL = "https://maps.googleapis.com/maps/api/streetview"
STREETVIEW_META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

_TIMEOUT = 30


@dataclass
class RetrievalResult:
    """Outcome of retrieving one site's ground-level image."""

    station_id: int | None
    source: str  # "place_photo" | "street_view" | "none"
    image_path: str | None
    place_id: str | None
    display_name: str | None
    resolved_lat: float | None
    resolved_lon: float | None
    error: str | None = None


def _require_key() -> str:
    if not config.MAPS_API_KEY:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY is not set. Add it to .env (see .env.example) and "
            "enable Places API (New) + Street View Static API on the key.\n"
            "This is separate from the Vertex AI ADC used by the classifier."
        )
    return config.MAPS_API_KEY


# --- Step 1: Nearby Search --------------------------------------------------
def nearby_ev_charger(
    lat: float, lon: float, radius_m: float | None = None
) -> dict | None:
    """Return the nearest EV charger place within ``radius_m``, or None.

    The returned dict holds ``id``, ``displayName``, ``location``, and (when
    present) ``photos`` — per the field mask below.
    """
    key = _require_key()
    radius = config.NEARBY_RADIUS_M if radius_m is None else radius_m
    body = {
        "includedTypes": ["electric_vehicle_charging_station"],
        "maxResultCount": 1,
        "rankPreference": "DISTANCE",
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": radius,
            }
        },
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.location,places.photos",
    }
    resp = requests.post(PLACES_NEARBY_URL, json=body, headers=headers, timeout=_TIMEOUT)
    resp.raise_for_status()
    places = resp.json().get("places", [])
    return places[0] if places else None


# --- Step 2: Place Photos ---------------------------------------------------
def place_photo_names(place_id: str) -> list[str]:
    """Return the photo resource names for a place via Place Details (New).

    Uses a stored ``place_id`` (e.g. ``ChIJ...``) so we fetch THE SAME place's
    full photo list without re-running a proximity search.
    """
    key = _require_key()
    url = PLACES_DETAILS_URL.format(place_id=place_id)
    headers = {"X-Goog-Api-Key": key, "X-Goog-FieldMask": "photos"}
    resp = requests.get(url, headers=headers, timeout=_TIMEOUT)
    resp.raise_for_status()
    return [p["name"] for p in resp.json().get("photos", [])]


def download_place_photo(
    photo_name: str, dest: Path, max_px: int | None = None
) -> Path:
    """Download a Place Photo (``places/{id}/photos/{id}``) to ``dest`` as JPEG."""
    key = _require_key()
    px = config.PHOTO_MAX_PX if max_px is None else max_px
    url = PLACES_PHOTO_URL.format(name=photo_name)
    params = {"maxWidthPx": px, "maxHeightPx": px, "key": key}
    resp = requests.get(url, params=params, timeout=_TIMEOUT * 2)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


# --- Step 3: Street View (with free metadata gate) --------------------------
def streetview_has_imagery(lat: float, lon: float) -> bool:
    """Free metadata check: True if Street View imagery exists at ``(lat, lon)``."""
    key = _require_key()
    params = {"location": f"{lat},{lon}", "key": key}
    resp = requests.get(STREETVIEW_META_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("status") == "OK"


def download_streetview(
    lat: float,
    lon: float,
    dest: Path,
    size: str | None = None,
    fov: int | None = None,
    heading: int | None = None,
    pitch: int | None = None,
) -> Path:
    """Download a Street View Static capture at ``(lat, lon)`` to ``dest``."""
    key = _require_key()
    params = {
        "location": f"{lat},{lon}",
        "size": size or config.STREETVIEW_SIZE,
        "fov": fov if fov is not None else config.STREETVIEW_FOV,
        "key": key,
    }
    if heading is not None:
        params["heading"] = heading
    if pitch is not None:
        params["pitch"] = pitch
    resp = requests.get(STREETVIEW_URL, params=params, timeout=_TIMEOUT * 2)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


# --- Orchestration ----------------------------------------------------------
def retrieve_for_site(
    lat: float,
    lon: float,
    station_id: int | None = None,
    out_dir: Path | None = None,
    radius_m: float | None = None,
) -> RetrievalResult:
    """Run the full dual-source pipeline for one coordinate.

    Branching (see NEXT_PLAN.md): no charger -> ``none``; charger with photos ->
    ``place_photo``; charger without photos but with Street View -> ``street_view``.
    """
    out_dir = out_dir or config.GROUND_IMAGES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = "site" if station_id is None else str(station_id)

    place = nearby_ev_charger(lat, lon, radius_m)
    if place is None:
        return RetrievalResult(
            station_id, "none", None, None, None, None, None,
            error="No EV charger within the threshold radius",
        )

    place_id = place.get("id")
    display_name = (place.get("displayName") or {}).get("text")
    loc = place.get("location") or {}
    rlat = loc.get("latitude", lat)
    rlon = loc.get("longitude", lon)

    # Route A: user-contributed photo (preferred).
    photos = place.get("photos") or []
    if photos:
        dest = out_dir / f"{sid}_place_photo.jpg"
        download_place_photo(photos[0]["name"], dest)
        return RetrievalResult(
            station_id, "place_photo", str(dest), place_id, display_name, rlat, rlon
        )

    # Route B: Street View fallback at the RESOLVED coordinate, metadata-gated.
    if streetview_has_imagery(rlat, rlon):
        dest = out_dir / f"{sid}_street_view.jpg"
        download_streetview(rlat, rlon, dest)
        return RetrievalResult(
            station_id, "street_view", str(dest), place_id, display_name, rlat, rlon
        )

    return RetrievalResult(
        station_id, "none", None, place_id, display_name, rlat, rlon,
        error="Charger found but no Place photo and no Street View imagery",
    )


def retrieve_with_retries(
    lat: float,
    lon: float,
    station_id: int | None = None,
    out_dir: Path | None = None,
    radius_m: float | None = None,
    max_retries: int = config.MAX_RETRIES,
) -> RetrievalResult:
    """``retrieve_for_site`` with exponential backoff on transient HTTP errors."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return retrieve_for_site(lat, lon, station_id, out_dir, radius_m)
        except requests.RequestException as e:  # transient network/API error
            last_err = e
            time.sleep(min(2**attempt, 16))
    return RetrievalResult(
        station_id, "none", None, None, None, None, None,
        error=f"request failed after {max_retries} retries: {str(last_err)[:200]}",
    )


if __name__ == "__main__":
    import json

    from .data import load_sites

    if not config.MAPS_API_KEY:
        print(
            "GOOGLE_MAPS_API_KEY is not set — cannot call the live APIs.\n"
            "Setup:\n"
            "  1. Create a Maps Platform API key in the Cloud console.\n"
            "  2. Enable: Places API (New) + Street View Static API.\n"
            "  3. Add to .env:  GOOGLE_MAPS_API_KEY=your_key_here\n"
            "Then re-run:  python -m src.retrieve_ground"
        )
        raise SystemExit(0)

    # Smoke test: retrieve the ground-level image for the first dataset site.
    row = load_sites().iloc[0]
    print(f"[smoke] station {row.station_id}: {row.latitude}, {row.longitude}")
    result = retrieve_with_retries(
        float(row.latitude), float(row.longitude), int(row.station_id)
    )
    print(json.dumps(asdict(result), indent=2))
