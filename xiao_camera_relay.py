#!/usr/bin/env python3
"""
XIAO ESP32S3 camera relay for multiple viewers

Upstream:
  http://<xiao-ip>:81/video   (MJPEG from XIAO)

Serves:
  /              -> home page
  /camera        -> camera page
  /camera-feed   -> relayed MJPEG stream
  /health        -> JSON health info

Why use this:
- only ONE connection goes to the XIAO
- many screens can connect to this Python server
- lower load on the XIAO than direct multi-viewer access
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from flask import Flask, Response, jsonify


@dataclass
class RelayState:
    connected: bool = False
    last_frame_time: float = 0.0
    frames_received: int = 0
    bytes_received: int = 0
    last_error: str = ""
    width_hint: int = 0
    height_hint: int = 0

    # FPS tracking
    start_time: float = field(default_factory=time.time)
    fps: float = 0.0
    fps_window_start: float = field(default_factory=time.time)
    fps_window_frames: int = 0


STATE = RelayState()
STATE_LOCK = threading.Lock()

LATEST_FRAME: Optional[bytes] = None
FRAME_COND = threading.Condition()


class XiaoCameraRelay(threading.Thread):
    def __init__(self, xiao_ip: str):
        super().__init__(daemon=True)
        self.url = f"http://{xiao_ip}:81/video"
        self.running = True

    def run(self):
        global LATEST_FRAME

        print(f"[relay] connecting to {self.url}")

        while self.running:
            try:
                with requests.get(self.url, stream=True, timeout=(5, 60)) as resp:
                    resp.raise_for_status()

                    with STATE_LOCK:
                        STATE.connected = True
                        STATE.last_error = ""

                    buffer = bytearray()

                    for chunk in resp.iter_content(chunk_size=4096):
                        if not self.running:
                            break
                        if not chunk:
                            continue

                        with STATE_LOCK:
                            STATE.bytes_received += len(chunk)

                        buffer.extend(chunk)

                        while True:
                            soi = buffer.find(b"\xff\xd8")  # JPEG start
                            if soi < 0:
                                # keep buffer from growing forever
                                if len(buffer) > 1024 * 1024:
                                    del buffer[:-4096]
                                break

                            eoi = buffer.find(b"\xff\xd9", soi + 2)  # JPEG end
                            if eoi < 0:
                                # wait for more bytes
                                if soi > 0:
                                    del buffer[:soi]
                                break

                            frame = bytes(buffer[soi:eoi + 2])
                            del buffer[:eoi + 2]

                            with FRAME_COND:
                                LATEST_FRAME = frame
                                FRAME_COND.notify_all()

                            now = time.time()
                            with STATE_LOCK:
                                STATE.frames_received += 1
                                STATE.last_frame_time = now
                                STATE.fps_window_frames += 1

                                elapsed = now - STATE.fps_window_start
                                if elapsed >= 1.0:
                                    STATE.fps = STATE.fps_window_frames / elapsed
                                    STATE.fps_window_start = now
                                    STATE.fps_window_frames = 0

            except Exception as e:
                with STATE_LOCK:
                    STATE.connected = False
                    STATE.last_error = str(e)
                print(f"[relay] error: {e}")
                time.sleep(2)


def create_app(xiao_ip: str) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def home():
        return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>XIAO Camera Relay</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      background: #111;
      color: #eee;
      padding: 24px;
    }}
    .card {{
      background: #1b1b1b;
      border-radius: 12px;
      padding: 18px;
      max-width: 760px;
    }}
    a {{ color: #9fd3ff; display:block; margin:10px 0; font-size:18px; }}
    code {{ background:#222; padding:2px 6px; border-radius:6px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>XIAO Camera Relay</h1>
    <a href="/camera">Open camera page</a>
    <a href="/camera-feed" target="_blank">Open raw relayed MJPEG</a>
    <a href="/health">Health JSON</a>
    <p>Upstream XIAO stream: <code>http://{xiao_ip}:81/video</code></p>
    <p>Share this page with viewers using your computer's IP, for example:
    <code>http://YOUR-PC-IP:5051/camera</code></p>
  </div>
</body>
</html>
"""

    @app.route("/camera")
    def camera_page():
        return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Camera Relay</title>
  <style>
    html, body {
      height: 100%;
      margin: 0;
      background: #111;
      color: #eee;
      font-family: Arial, sans-serif;
    }
    body {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
      box-sizing: border-box;
    }
    .card {
      background: #1b1b1b;
      border-radius: 12px;
      padding: 16px;
      width: min(96vw, 1100px);
      text-align: center;
    }
    img {
      width: 100%;
      max-height: 80vh;
      object-fit: contain;
      border-radius: 8px;
      display: block;
      margin: 0 auto;
      background: #000;
    }
    .meta {
      margin-top: 12px;
      color: #bbb;
      font-size: 14px;
    }
    a { color: #9fd3ff; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Camera video</h1>
    <img src="/camera-feed" alt="camera relay">
    <div class="meta">
      <a href="/camera-feed" target="_blank">Open raw relayed MJPEG</a>
    </div>
  </div>
</body>
</html>
"""

    @app.route("/camera-feed")
    def camera_feed():
        def generate():
            last_sent = None

            while True:
                with FRAME_COND:
                    if LATEST_FRAME is None:
                        FRAME_COND.wait(timeout=5.0)
                    else:
                        FRAME_COND.wait(timeout=1.0)

                    frame = LATEST_FRAME

                if frame is None:
                    continue

                # avoid resending identical object too aggressively if nothing updated
                if frame is last_sent:
                    time.sleep(0.03)
                    continue
                last_sent = frame

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                    frame + b"\r\n"
                )

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/health")
    def health():
        with STATE_LOCK:
            now = time.time()
            age = now - STATE.last_frame_time if STATE.last_frame_time else None
            uptime = now - STATE.start_time
            avg_fps = (STATE.frames_received / uptime) if uptime > 0 else 0.0

            return jsonify({
                "connected_to_xiao": STATE.connected,
                "upstream": f"http://{xiao_ip}:81/video",
                "frames_received": STATE.frames_received,
                "bytes_received": STATE.bytes_received,
                "last_frame_age_sec": round(age, 3) if age is not None else None,
                "last_error": STATE.last_error,
                "has_latest_frame": LATEST_FRAME is not None,
                "fps": round(STATE.fps, 2),          # recent live FPS
                "avg_fps": round(avg_fps, 2),        # average since startup
                "uptime_sec": round(uptime, 2),
            })

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xiao-ip", required=True, help="IP of the XIAO board")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5051)
    args = parser.parse_args()

    relay = XiaoCameraRelay(args.xiao_ip)
    relay.start()

    app = create_app(args.xiao_ip)
    print(f"[server] home:        http://127.0.0.1:{args.port}/")
    print(f"[server] camera:      http://127.0.0.1:{args.port}/camera")
    print(f"[server] camera-feed: http://127.0.0.1:{args.port}/camera-feed")
    print(f"[server] health:      http://127.0.0.1:{args.port}/health")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()