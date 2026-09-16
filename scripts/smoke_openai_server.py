#!/usr/bin/env python3
"""Smoke-test a vLLM or llama.cpp OpenAI-compatible local server."""

from __future__ import annotations

import argparse
import json
import time
from urllib.request import Request, urlopen


def get_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None, help="Served name; defaults to the first listed model")
    parser.add_argument("--wait-ready", type=float, default=0,
                        help="Seconds to allow for server startup before checking the endpoint.")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    deadline = time.monotonic() + args.wait_ready
    while True:
        try:
            models = get_json(f"{base}/models")
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    entries = models.get("data", [])
    if not entries:
        raise RuntimeError("Server returned no models")
    model = args.model or entries[0]["id"]
    result = post_json(
        f"{base}/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: READY"}],
            "temperature": 0,
            "max_tokens": 8,
        },
    )
    text = result["choices"][0]["message"]["content"]
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Server returned no visible completion")
    print(json.dumps({"server_model": model, "response": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
