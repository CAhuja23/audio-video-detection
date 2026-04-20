#!/usr/bin/env python3
"""
XIAO ESP32S3 Sense raw audio analysis server with MJPEG waveform feed

Reads:
  http://<xiao-ip>:82/stream/audio   (PCM16 mono 16kHz)

Serves:
  /               -> home page
  /video          -> waveform-only MJPEG page
  /waveform-feed  -> raw waveform MJPEG stream
  /analysis       -> waveform + labels
  /both           -> camera video + waveform MJPEG + labels
  /api/latest     -> JSON status

What it classifies:
  - high-level audio event class: person / vehicle / animal / object / other
  - top YAMNet labels
  - language via Whisper when speech-like audio is present
"""

from __future__ import annotations

import argparse
import io
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests
from flask import Flask, Response, jsonify
from scipy.io import wavfile
from PIL import Image, ImageDraw

import librosa
import tensorflow as tf
import pandas as pd

try:
    import whisper as whisper_lib
except ImportError as e:
    raise RuntimeError("Install Whisper: pip install openai-whisper") from e


# =========================
# Config
# =========================

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

CHUNK_SECONDS = 2.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
WHISPER_MODEL = "tiny"
WAVEFORM_SECONDS = 8
WAVEFORM_SAMPLES = SAMPLE_RATE * WAVEFORM_SECONDS
MODEL_BASE = Path("local_models")
WHISPER_DIR = MODEL_BASE / "whisper"
YAMNET_MODEL_HANDLE = MODEL_BASE / "yamnet_saved_model"
YAMNET_CLASS_MAP_CSV = MODEL_BASE / "yamnet_class_map.csv"

PRESENCE_THRESHOLD = 0.20
MULTILABEL_THRESHOLD = 0.30

AUDIO_SAVE_DIR = Path("saved_audio")
AUDIO_SAVE_DIR.mkdir(exist_ok=True)


# =========================
# Label groups
# =========================

PERSON_LABELS = {
    "Speech", "Conversation", "Narration, monologue", "Babbling",
    "Child speech, kid speaking", "Shout", "Screaming", "Singing",
    "Yell", "Chatter", "Crowd", "Whispering", "Laughter",
}
VEHICLE_LABELS = {
    "Vehicle", "Car", "Truck", "Motorcycle", "Bus", "Emergency vehicle",
    "Siren", "Engine", "Idling", "Car passing by", "Rail transport",
    "Train", "Aircraft", "Helicopter", "Motorboat, speedboat",
}
ANIMAL_LABELS = {
    "Animal", "Dog", "Bark", "Growling", "Cat", "Meow", "Bird",
    "Bird vocalization, bird call, bird song", "Insect", "Horse",
    "Roar", "Frog", "Snake", "Cattle, bovine", "Pig", "Sheep",
}
OBJECT_LABELS = {
    "Door", "Doorbell", "Knock", "Tool", "Hammer", "Sawing", "Drill",
    "Alarm", "Beep", "Glass", "Glass breaking", "Cutlery, silverware",
    "Computer keyboard", "Typing", "Footsteps", "Gunshot, gunfire", "Explosion",
}


def map_high_level(label: str) -> str:
    if label in PERSON_LABELS:
        return "person"
    if label in VEHICLE_LABELS:
        return "vehicle"
    if label in ANIMAL_LABELS:
        return "animal"
    if label in OBJECT_LABELS:
        return "object"
    return "other"


# =========================
# Models
# =========================

class YamnetWrapper:
    def __init__(self, model_dir: Path, class_map_csv: Path):
        if not model_dir.exists():
            raise RuntimeError(f"Missing YAMNet model dir: {model_dir}")
        if not class_map_csv.exists():
            raise RuntimeError(f"Missing YAMNet class map: {class_map_csv}")

        print(f"[yamnet] loading model from {model_dir}")
        self.model = tf.saved_model.load(str(model_dir))

        df = pd.read_csv(class_map_csv)
        self.class_names = df["display_name"].tolist()
        print(f"[yamnet] loaded {len(self.class_names)} classes")

    def classify_chunk(self, chunk_int16: np.ndarray, sr: int) -> dict:
        x = chunk_int16.astype(np.float32) / 32768.0
        if sr != 16000:
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)

        waveform = tf.convert_to_tensor(x, dtype=tf.float32)
        scores, _, _ = self.model(waveform)
        mean_scores = scores.numpy().mean(axis=0)

        presence_score = float(mean_scores.max())
        detected: List[Tuple[str, float]] = []
        for prob, name in zip(mean_scores, self.class_names):
            if prob >= MULTILABEL_THRESHOLD:
                detected.append((name, float(prob)))

        detected.sort(key=lambda t: -t[1])
        top_raw = detected[:5]

        high_level = {
            "person":  {"present": False, "score": 0.0},
            "vehicle": {"present": False, "score": 0.0},
            "animal":  {"present": False, "score": 0.0},
            "object":  {"present": False, "score": 0.0},
            "other":   {"present": False, "score": 0.0},
        }

        for name, prob in detected:
            cat = map_high_level(name)
            high_level[cat]["present"] = True
            high_level[cat]["score"] = max(high_level[cat]["score"], prob)

        best_category = "none"
        best_score = 0.0
        for cat, info in high_level.items():
            if info["score"] > best_score:
                best_category = cat
                best_score = info["score"]

        return {
            "presence_score": presence_score,
            "top_raw_labels": top_raw,
            "high_level_summary": high_level,
            "best_category": best_category if presence_score >= PRESENCE_THRESHOLD else "none",
        }


class WhisperLanguageDetector:
    def __init__(self, model_name: str = "tiny"):
        print(f"[whisper] loading model: {model_name}")
        self.model = whisper_lib.load_model(
                WHISPER_MODEL,
                download_root=str(WHISPER_DIR),
            )

    def detect_language(self, chunk_int16: np.ndarray, sr: int) -> Optional[str]:
        try:
            audio_float = chunk_int16.astype(np.float32) / 32768.0
            audio_float = audio_float - np.mean(audio_float)
            if sr != 16000:
                audio_float = librosa.resample(audio_float, orig_sr=sr, target_sr=16000)

            result = self.model.transcribe(audio_float, task="transcribe", language=None, verbose=False)
            return result.get("language")
        except Exception as e:
            print(f"[whisper] detect_language error: {e}")
            return None


# =========================
# Shared state
# =========================

@dataclass
class AnalysisState:
    latest_chunk_time: float = 0.0
    status: str = "starting"
    level_rms: float = 0.0
    level_peak: float = 0.0
    event_class: str = "unknown"
    language: str = "unknown"
    summary: str = "waiting for audio"
    top_labels: List[Tuple[str, float]] = field(default_factory=list)
    waveform: deque = field(default_factory=lambda: deque(maxlen=WAVEFORM_SAMPLES))
    last_audio_file: Optional[str] = None


STATE = AnalysisState()
STATE_LOCK = threading.Lock()


# =========================
# Helpers
# =========================

def rms_and_peak(samples: np.ndarray) -> Tuple[float, float]:
    if samples.size == 0:
        return 0.0, 0.0
    x = samples.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    return rms, peak


def make_summary(event_class: str, language: str, top_labels: List[Tuple[str, float]]) -> str:
    if event_class == "none":
        return "No strong audio event detected."
    top_text = ", ".join(lbl for lbl, _ in top_labels[:2]) if top_labels else "unknown"
    if event_class == "person":
        if language and language != "unknown":
            return f"Speech/person-like audio detected. Language appears to be {language}. Top sounds: {top_text}."
        return f"Speech/person-like audio detected. Top sounds: {top_text}."
    return f"{event_class.capitalize()}-like audio detected. Top sounds: {top_text}."


def save_wav(chunk_int16: np.ndarray) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = AUDIO_SAVE_DIR / f"xiao_audio_{ts}.wav"
    wavfile.write(str(path), SAMPLE_RATE, chunk_int16)
    return path.name


def render_waveform_jpeg(
    samples: List[int],
    width: int = 1100,
    height: int = 420,
    lang: str = "UNKNOWN",
    event: str = "UNKNOWN",
    status: str = "UNKNOWN",
) -> bytes:
    img = Image.new("RGB", (width, height), (6, 9, 7))
    draw = ImageDraw.Draw(img)

    mid = height // 2
    draw.line((0, mid, width, mid), fill=(30, 50, 35), width=1)

    if samples:
        step = max(1, len(samples) // width)
        points = []
        x = 0
        gain = 4.5

        for i in range(0, len(samples), step):
            if x >= width:
                break

            amp = (samples[i] / 32768.0) * gain
            amp = max(-1.0, min(1.0, amp))
            y = int(mid - amp * (height * 0.42))
            points.append((x, y))
            x += 1

        if len(points) > 1:
            draw.line(points, fill=(121, 255, 156), width=3)

    # ----- info boxes drawn into image -----
    def draw_box(x, y, title, value, w=210, h=70):
        draw.rounded_rectangle(
            (x, y, x + w, y + h),
            radius=4,
            fill=(9, 13, 10),
            outline=(30, 64, 39),
            width=1
        )
        draw.text((x + 12, y + 10), title, fill=(107, 142, 116))
        draw.text((x + 12, y + 34), value, fill=(121, 255, 156))

    draw_box(20, 20, "LANG", str(lang).upper())
    draw_box(250, 20, "EVENT", str(event).upper())
    draw_box(480, 20, "STATUS", str(status).upper())

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def waveform_downsample(samples: List[int], max_points: int = 1200) -> List[int]:
    if len(samples) <= max_points:
        return samples
    idx = np.linspace(0, len(samples) - 1, max_points).astype(int)
    return [samples[i] for i in idx]


# =========================
# Audio reader / analyzer
# =========================

class XiaoAudioAnalyzer(threading.Thread):
    def __init__(self, xiao_ip: str, yamnet_model_dir: Path, yamnet_csv: Path, whisper_model: str):
        super().__init__(daemon=True)
        self.stream_url = f"http://{xiao_ip}:82/stream/audio"
        self.yamnet = YamnetWrapper(yamnet_model_dir, yamnet_csv)
        self.whisper = WhisperLanguageDetector(whisper_model)
        self.running = True

    def run(self):
        print(f"[audio] connecting to {self.stream_url}")
        while self.running:
            try:
                with requests.get(self.stream_url, stream=True, timeout=(5, 30)) as resp:
                    resp.raise_for_status()
                    buffer = bytearray()

                    with STATE_LOCK:
                        STATE.status = "connected"

                    for chunk in resp.iter_content(chunk_size=2048):
                        if not self.running:
                            break
                        if not chunk:
                            continue

                        buffer.extend(chunk)

                        usable = chunk[: len(chunk) - (len(chunk) % 2)]
                        pcm_now = np.frombuffer(usable, dtype="<i2").astype(np.int16)
                        if pcm_now.size > 0:
                            with STATE_LOCK:
                                STATE.waveform.extend(pcm_now.tolist())

                        need_bytes = CHUNK_SAMPLES * BYTES_PER_SAMPLE
                        while len(buffer) >= need_bytes:
                            raw = bytes(buffer[:need_bytes])
                            del buffer[:need_bytes]
                            pcm = np.frombuffer(raw, dtype="<i2").astype(np.int16)
                            self.analyze_pcm_chunk(pcm)

            except Exception as e:
                print(f"[audio] stream error: {e}")
                with STATE_LOCK:
                    STATE.status = "reconnecting"
                time.sleep(2)

    def analyze_pcm_chunk(self, pcm: np.ndarray):
        rms, peak = rms_and_peak(pcm)
        yres = self.yamnet.classify_chunk(pcm, SAMPLE_RATE)

        language = "unknown"
        if yres["best_category"] == "person":
            detected = self.whisper.detect_language(pcm, SAMPLE_RATE)
            if detected:
                language = detected

        summary = make_summary(yres["best_category"], language, yres["top_raw_labels"])
        saved_name = save_wav(pcm)

        with STATE_LOCK:
            STATE.latest_chunk_time = time.time()
            STATE.status = "analyzed"
            STATE.level_rms = rms
            STATE.level_peak = peak
            STATE.event_class = yres["best_category"]
            STATE.language = language
            STATE.summary = summary
            STATE.top_labels = yres["top_raw_labels"]
            STATE.last_audio_file = saved_name


# =========================
# Flask app
# =========================

def create_app(xiao_ip: str) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def home():
        return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>XIAO Dashboard</title>
  <style>
    body {{ font-family: 'Inter', system-ui, sans-serif; background: #060907; color: #d6e4da; padding: 24px; }}
    .card {{ background:#090d0a; padding:20px; border-radius:4px; max-width:700px; border:1px solid rgba(121,255,156,0.18); }}
    h1 {{ color:#79ff9c; font-size:1.1rem; text-transform:uppercase; letter-spacing:0.14em; text-shadow:0 0 10px rgba(121,255,156,0.4); }}
    a {{ color:#79ff9c; display:block; margin:10px 0; font-size:0.88rem; text-transform:uppercase; letter-spacing:0.1em; text-decoration:none; opacity:0.8; transition:opacity 0.15s; }}
    a:hover {{ opacity:1; }}
    code {{ background:rgba(121,255,156,0.08); color:#79ff9c; padding:2px 6px; border-radius:2px; font-size:0.82rem; }}
    p {{ color:rgba(214,228,218,0.5); font-size:0.82rem; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>XIAO Dashboard</h1>
    <a href="/video">Waveform only</a>
    <a href="/analysis">Waveform + classification</a>
    <a href="/both">Camera video + waveform + classification</a>
    <a href="/waveform-feed">Raw waveform MJPEG</a>
    <a href="/api/latest">Raw JSON</a>
    <p>XIAO video stream: <code>http://{xiao_ip}:81/video</code></p>
    <p>XIAO audio stream: <code>http://{xiao_ip}:82/stream/audio</code></p>
  </div>
</body>
</html>
"""

    @app.route("/api/latest")
    def api_latest():
        with STATE_LOCK:
            waveform = waveform_downsample(list(STATE.waveform))
            return jsonify({
                "status": STATE.status,
                "latest_chunk_time": STATE.latest_chunk_time,
                "level_rms": STATE.level_rms,
                "level_peak": STATE.level_peak,
                "event_class": STATE.event_class,
                "language": STATE.language,
                "summary": STATE.summary,
                "top_labels": STATE.top_labels,
                "waveform": waveform,
                "last_audio_file": STATE.last_audio_file,
                "video_url": f"http://{xiao_ip}:81/video",
                "audio_raw_url": f"http://{xiao_ip}:82/stream/audio",
                "waveform_feed_url": "/waveform-feed",
            })

    @app.route("/audio/<filename>")
    def serve_audio(filename: str):
        path = AUDIO_SAVE_DIR / filename
        if not path.exists():
            return ("not found", 404)
        return Response(
            path.read_bytes(),
            mimetype="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )


    @app.route("/waveform-feed")
    def waveform_feed():
        def generate():
              while True:
                  with STATE_LOCK:
                      samples = waveform_downsample(list(STATE.waveform))
                      lang = STATE.language if STATE.language else "unknown"
                      event = STATE.event_class if STATE.event_class else "unknown"
                      status = STATE.status if STATE.status else "unknown"

                  jpg = render_waveform_jpeg(
                      samples,
                      width=1100,
                      height=420,
                      lang=lang,
                      event=event,
                      status=status,
                  )

                  yield (
                      b"--frame\r\n"
                      b"Content-Type: image/jpeg\r\n"
                      b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                      jpg + b"\r\n"
                  )

                  time.sleep(0.1)
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/video")
    def video_page():
          return """
      <!doctype html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>Waveform Video</title>
        <style>
          html, body { height:100%; margin:0; }
          body { font-family:'Inter',system-ui,sans-serif; background:#060907; color:#d6e4da; display:flex; align-items:center; justify-content:center; }
          .card { background:#090d0a; border:1px solid rgba(121,255,156,0.18); border-radius:4px; padding:16px; width:min(92vw,960px); text-align:center; }
          h1 { color:#79ff9c; font-size:0.9rem; text-transform:uppercase; letter-spacing:0.14em; text-shadow:0 0 10px rgba(121,255,156,0.4); }
          img { width:100%; max-height:70vh; object-fit:contain; border-radius:2px; display:block; margin:0 auto; background:#000; }
          a { color:#79ff9c; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.1em; opacity:0.7; }
          a:hover { opacity:1; }
        </style>
      </head>
      <body>
        <div class="card">
          <h1>Waveform video feed</h1>
          <img src="/waveform-feed" alt="waveform feed">
          <p><a href="/waveform-feed" target="_blank">Open raw waveform MJPEG feed</a></p>
        </div>
      </body>
      </html>
      """
    @app.route("/analysis")
    def analysis_page():
          return """
  <!doctype html>
  <html>
  <head>
    <meta charset="utf-8">
    <title>Audio Analysis</title>
    <style>
      body { font-family:'Inter',system-ui,sans-serif; background:#060907; color:#d6e4da; padding:20px; }
      .card { background:#090d0a; border:1px solid rgba(121,255,156,0.18); border-radius:4px; padding:16px; max-width:900px; }
      h1 { color:#79ff9c; font-size:0.9rem; text-transform:uppercase; letter-spacing:0.14em; text-shadow:0 0 10px rgba(121,255,156,0.4); }
      canvas { background:#000; border-radius:2px; display:block; margin-top:12px; }
      .grid { display:grid; grid-template-columns:180px 1fr; gap:8px 12px; margin-top:16px; font-size:0.82rem; }
      .label { color:rgba(214,228,218,0.5); text-transform:uppercase; letter-spacing:0.08em; font-size:0.72rem; }
      .pill { display:inline-block; background:rgba(121,255,156,0.08); color:#79ff9c; padding:3px 8px; border-radius:2px; margin:4px 6px 0 0; font-size:0.75rem; border:1px solid rgba(121,255,156,0.18); }
      a { color:#79ff9c; font-size:0.78rem; text-decoration:none; opacity:0.7; }
      a:hover { opacity:1; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Audio analysis</h1>
      <canvas id="wave" width="820" height="160"></canvas>
      <div class="grid">
        <div class="label">Status</div><div id="status">-</div>
        <div class="label">Event</div><div id="event">-</div>
        <div class="label">Language</div><div id="lang">-</div>
        <div class="label">RMS</div><div id="rms">-</div>
        <div class="label">Peak</div><div id="peak">-</div>
        <div class="label">Summary</div><div id="summary">-</div>
        <div class="label">Top labels</div><div id="labels">-</div>
        <div class="label">Audio file</div><div id="audiofile">-</div>
      </div>
    </div>

    <script>
      const canvas = document.getElementById("wave");
      const ctx = canvas.getContext("2d");

      function drawWave(samples) {
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const mid = canvas.height / 2;
        ctx.strokeStyle = "#79ff9c";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, mid);

        if (!samples || samples.length === 0) {
          ctx.lineTo(canvas.width, mid);
          ctx.stroke();
          return;
        }

        const step = Math.max(1, Math.floor(samples.length / canvas.width));
        let x = 0;
        for (let i = 0; i < samples.length && x < canvas.width; i += step) {
          const y = mid - (samples[i] / 32768.0) * (mid * 0.9);
          ctx.lineTo(x, y);
          x++;
        }
        ctx.stroke();
      }

      async function refresh() {
        const r = await fetch('/api/latest');
        const d = await r.json();

        document.getElementById('status').textContent = d.status;
        document.getElementById('event').textContent = d.event_class;
        document.getElementById('lang').textContent = d.language;
        document.getElementById('rms').textContent = Number(d.level_rms || 0).toFixed(3);
        document.getElementById('peak').textContent = Number(d.level_peak || 0).toFixed(3);
        document.getElementById('summary').textContent = d.summary || '-';

        const labels = document.getElementById('labels');
        labels.innerHTML = '';
        (d.top_labels || []).forEach(item => {
          const el = document.createElement('span');
          el.className = 'pill';
          el.textContent = `${item[0]} (${Number(item[1]).toFixed(2)})`;
          labels.appendChild(el);
        });

        const audiofile = document.getElementById('audiofile');
        if (d.last_audio_file) {
          audiofile.innerHTML = `<a href="/audio/${d.last_audio_file}">Download latest WAV</a>`;
        } else {
          audiofile.textContent = '-';
        }

        drawWave(d.waveform || []);
      }

      refresh();
      setInterval(refresh, 1000);
    </script>
  </body>
  </html>
  """
    @app.route("/waveform-info")
    def waveform_info_page():
          return """
      <!doctype html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>Waveform + Info</title>
        <style>
          html, body { height:100%; margin:0; background:#060907; color:#d6e4da; font-family:'Inter',system-ui,sans-serif; }
          body { display:flex; align-items:center; justify-content:center; padding:20px; box-sizing:border-box; }
          .card { background:#090d0a; border:1px solid rgba(121,255,156,0.18); border-radius:4px; padding:16px; width:min(96vw,1250px); }
          .frame { position:relative; width:100%; background:#000; border-radius:2px; overflow:hidden; }
          .frame img { width:100%; display:block; border-radius:2px; }
          .overlay { position:absolute; left:16px; top:16px; display:flex; flex-wrap:wrap; gap:10px; pointer-events:none; }
          .box { background:rgba(6,9,7,0.8); border:1px solid rgba(121,255,156,0.18); border-radius:2px; padding:10px 14px; min-width:150px; backdrop-filter:blur(3px); }
          .k { font-size:0.68rem; color:rgba(214,228,218,0.5); margin-bottom:4px; letter-spacing:0.1em; text-transform:uppercase; }
          .v { font-size:1.1rem; font-weight:700; color:#79ff9c; text-shadow:0 0 8px rgba(121,255,156,0.3); }
          .sub { margin-top:12px; color:rgba(214,228,218,0.5); font-size:0.82rem; }
          a { color:#79ff9c; font-size:0.78rem; opacity:0.7; }
          a:hover { opacity:1; }
        </style>
      </head>
      <body>
        <div class="card">

          <div class="frame">
            <img src="/waveform-feed" alt="waveform feed">

            <div class="overlay">
              <div class="box">
                <div class="k">LANG</div>
                <div class="v" id="langBox">--</div>
              </div>

              <div class="box">
                <div class="k">EVENT</div>
                <div class="v" id="eventBox">--</div>
              </div>

            </div>
          </div>
        </div>

        <script>
          function fmtUpper(v, fallback="--") {
            if (!v) return fallback;
            return String(v).toUpperCase();
          }

          async function refreshInfo() {
            try {
              const r = await fetch('/api/latest');
              const d = await r.json();

              document.getElementById('langBox').textContent =
                d.language && d.language !== 'unknown' ? fmtUpper(d.language) : 'UNKNOWN';

              document.getElementById('eventBox').textContent =
                d.event_class ? fmtUpper(d.event_class) : '--';

              document.getElementById('statusBox').textContent =
                d.status ? fmtUpper(d.status) : '--';

              document.getElementById('summaryBox').textContent = d.summary || '-';

              const labels = (d.top_labels || []).map(x => `${x[0]} (${Number(x[1]).toFixed(2)})`);
              document.getElementById('labelsBox').textContent = labels.length ? labels.join(', ') : '-';

              if (d.last_audio_file) {
                document.getElementById('audioBox').innerHTML =
                  `<a href="/audio/${d.last_audio_file}">Download latest WAV</a>`;
              } else {
                document.getElementById('audioBox').textContent = '-';
              }
            } catch (e) {
              document.getElementById('statusBox').textContent = 'ERROR';
            }
          }

          refreshInfo();
          setInterval(refreshInfo, 1000);
        </script>
      </body>
      </html>
      """
    @app.route("/camera")
    def camera_page():
        return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>XIAO Camera</title>
  <style>
    html, body {{ height:100%; margin:0; background:#060907; color:#d6e4da; font-family:'Inter',system-ui,sans-serif; }}
    body {{ display:flex; align-items:center; justify-content:center; padding:20px; box-sizing:border-box; }}
    h1 {{ color:#79ff9c; font-size:0.9rem; text-transform:uppercase; letter-spacing:0.14em; text-shadow:0 0 10px rgba(121,255,156,0.4); }}
    .card {{ background:#090d0a; border:1px solid rgba(121,255,156,0.18); border-radius:4px; padding:16px; width:min(96vw,1100px); text-align:center; }}
    img {{ width:100%; max-height:80vh; object-fit:contain; border-radius:2px; display:block; margin:0 auto; background:#000; }}
    a {{ color:#79ff9c; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.1em; opacity:0.7; text-decoration:none; }}
    a:hover {{ opacity:1; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Camera video</h1>
    <img src="http://{xiao_ip}:81/video" alt="camera video">
    <p><a href="http://{xiao_ip}:81/video" target="_blank">Open raw XIAO camera stream</a></p>
  </div>
</body>
</html>
"""
    @app.route("/both")
    def both_page():
          return f"""
  <!doctype html>
  <html>
  <head>
    <meta charset="utf-8">
    <title>Video + Audio Analysis</title>
    <style>
      body {{ font-family:'Inter',system-ui,sans-serif; background:#060907; color:#d6e4da; padding:20px; }}
      .wrap {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
      .card {{ background:#090d0a; border:1px solid rgba(121,255,156,0.18); border-radius:4px; padding:16px; }}
      h2 {{ color:#79ff9c; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.14em; text-shadow:0 0 10px rgba(121,255,156,0.4); }}
      img {{ max-width:100%; border-radius:2px; display:block; }}
      .grid {{ display:grid; grid-template-columns:140px 1fr; gap:8px 12px; margin-top:16px; font-size:0.82rem; }}
      .label {{ color:rgba(214,228,218,0.5); text-transform:uppercase; letter-spacing:0.08em; font-size:0.72rem; }}
      .pill {{ display:inline-block; background:rgba(121,255,156,0.08); color:#79ff9c; padding:3px 8px; border-radius:2px; margin:4px 6px 0 0; font-size:0.75rem; border:1px solid rgba(121,255,156,0.18); }}
      a {{ color:#79ff9c; font-size:0.78rem; text-decoration:none; opacity:0.7; }}
      a:hover {{ opacity:1; }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <h2>Camera video</h2>
        <img src="http://{xiao_ip}:81/video" alt="camera video">
      </div>

      <div class="card">
        <h2>Waveform video</h2>
        <img src="/waveform-feed" alt="waveform feed">
        <div class="grid">
          <div class="label">Status</div><div id="status">-</div>
          <div class="label">Event</div><div id="event">-</div>
          <div class="label">Language</div><div id="lang">-</div>
          <div class="label">Summary</div><div id="summary">-</div>
          <div class="label">Top labels</div><div id="labels">-</div>
          <div class="label">Audio file</div><div id="audiofile">-</div>
        </div>
      </div>
    </div>

    <script>
      async function refresh() {{
        const r = await fetch('/api/latest');
        const d = await r.json();

        document.getElementById('status').textContent = d.status;
        document.getElementById('event').textContent = d.event_class;
        document.getElementById('lang').textContent = d.language;
        document.getElementById('summary').textContent = d.summary || '-';

        const labels = document.getElementById('labels');
        labels.innerHTML = '';
        (d.top_labels || []).forEach(item => {{
          const el = document.createElement('span');
          el.className = 'pill';
          el.textContent = `${{item[0]}} (${{Number(item[1]).toFixed(2)}})`;
          labels.appendChild(el);
        }});

        const audiofile = document.getElementById('audiofile');
        if (d.last_audio_file) {{
          audiofile.innerHTML = `<a href="/audio/${{d.last_audio_file}}">Download latest WAV</a>`;
        }} else {{
          audiofile.textContent = '-';
        }}
      }}

      refresh();
      setInterval(refresh, 1000);
    </script>
  </body>
  </html>
  """

    return app


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xiao-ip", required=True, help="IP of the XIAO board")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--whisper-model", default="tiny")
    args = parser.parse_args()

    analyzer = XiaoAudioAnalyzer(
        xiao_ip=args.xiao_ip,
        yamnet_model_dir=YAMNET_MODEL_HANDLE,
        yamnet_csv=YAMNET_CLASS_MAP_CSV,
        whisper_model=args.whisper_model,
    )
    analyzer.start()

    app = create_app(args.xiao_ip)
    print(f"[server] root:          http://127.0.0.1:{args.port}/")
    print(f"[server] video:         http://127.0.0.1:{args.port}/video")
    print(f"[server] analysis:      http://127.0.0.1:{args.port}/analysis")
    print(f"[server] both:          http://127.0.0.1:{args.port}/both")
    print(f"[server] waveform-feed: http://127.0.0.1:{args.port}/waveform-feed")
    
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)



if __name__ == "__main__":
    main()