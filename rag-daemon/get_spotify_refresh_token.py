#!/usr/bin/env python3
"""
One-time script to obtain a Spotify refresh token for the RAG daemon.

Prerequisites:
  1. Create an app at https://developer.spotify.com/dashboard
  2. In the app, go to Settings → Redirect URIs and add exactly:
       http://127.0.0.1:8766/callback
     (Spotify requires 127.0.0.1, not "localhost", for local redirects.)
  3. Save, then set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env

Run:
  python get_spotify_refresh_token.py

Then paste the printed SPOTIFY_REFRESH_TOKEN into your .env.
"""

from __future__ import annotations

import os
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import requests

# Spotify allows http only for loopback IP (127.0.0.1), not "localhost". Must match Dashboard exactly.
REDIRECT_URI = os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8766/callback").strip()
SCOPES = "user-modify-playback-state user-read-playback-state"


def main() -> None:
    client_id = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        print("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env or the environment.")
        return

    auth_url = (
        "https://accounts.spotify.com/authorize?"
        f"client_id={client_id}&response_type=code&redirect_uri={REDIRECT_URI}&scope={SCOPES}"
    )

    result: dict = {"code": None, "error": None}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            if "code" in qs:
                result["code"] = qs["code"][0]
            if "error" in qs:
                result["error"] = qs["error"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><p>You can close this tab and return to the terminal.</p></body></html>"
            )

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 8766), Handler)
    print("Opening browser for Spotify authorization...")
    print("If it does not open, go to:", auth_url)
    webbrowser.open(auth_url)
    server.handle_request()
    server.server_close()

    if result["error"]:
        print("Authorization failed:", result["error"])
        return
    if not result["code"]:
        print("No code received. Did you approve the app?")
        return

    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "authorization_code",
            "code": result["code"],
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    refresh = data.get("refresh_token")
    if not refresh:
        print("Response did not contain refresh_token:", data.keys())
        return
    print("\nAdd this to your .env:\n")
    print(f"SPOTIFY_REFRESH_TOKEN={refresh}\n")


if __name__ == "__main__":
    main()
