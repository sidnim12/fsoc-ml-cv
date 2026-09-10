from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import httpx


DEFAULT_API_URL = "http://127.0.0.1:8000"


def post_frame(api_url: str, image_path: Path, frame_index: int | None, session_id: str) -> Dict[str, Any]:
    """Send one image to the running FastAPI service and return JSON."""
    if not image_path.exists():
        raise FileNotFoundError(f"image does not exist: {image_path}")
    with image_path.open("rb") as handle:
        files = {"file": (image_path.name, handle, "image/png")}
        data: Dict[str, str] = {"session_id": session_id}
        if frame_index is not None:
            data["frame_index"] = str(frame_index)
        response = httpx.post(f"{api_url.rstrip('/')}/predict-frame", files=files, data=data, timeout=30.0)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("API response must be a JSON object")
    return payload


def health(api_url: str) -> Dict[str, Any]:
    """Read API health JSON."""
    response = httpx.get(f"{api_url.rstrip('/')}/health", timeout=10.0)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("health response must be a JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the FSOC FastAPI backend with one Unity frame.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--image", required=True, help="PNG/JPG frame to send to /predict-frame.")
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--session-id", default="smoke_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_url = str(args.api_url)
    print("Health:")
    print(json.dumps(health(api_url), indent=2))
    print("Prediction:")
    print(json.dumps(post_frame(api_url, Path(args.image), args.frame_index, args.session_id), indent=2))


if __name__ == "__main__":
    main()
