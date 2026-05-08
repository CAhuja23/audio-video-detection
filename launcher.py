#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import requests


# ============================================================
# HARDCODED CONFIG
# ============================================================

CANDIDATE_IPS = [
    "10.0.50.93",
    "10.0.50.121",
    "10.0.50.85",
]

AUDIO_SCRIPT = "xaio_server_audio.py"
CAMERA_SCRIPT = "xiao_camera_relay.py"

HOST = "0.0.0.0"

# Fixed port layout
CAM1_AUDIO_PORT = 6001
CAM1_CAMERA_PORT = 6002
CAM2_AUDIO_PORT = 6003
CAM2_CAMERA_PORT = 6004

MAX_CAMERAS = 2


# ============================================================
# Probe helpers
# ============================================================

def probe_stream_once(
    url: str,
    read_bytes: int = 1024,
    timeout_connect: float = 2.0,
    timeout_read: float = 3.0,
) -> bool:
    try:
        with requests.get(url, stream=True, timeout=(timeout_connect, timeout_read)) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=read_bytes):
                if chunk:
                    return True
        return False
    except Exception:
        return False


def probe_camera_ip(ip: str) -> Tuple[str, bool, str]:
    video_url = f"http://{ip}:81/video"
    audio_url = f"http://{ip}:82/stream/audio"

    video_ok = probe_stream_once(video_url)
    audio_ok = probe_stream_once(audio_url)

    if video_ok and audio_ok:
        return ip, True, "video+audio OK"
    if video_ok and not audio_ok:
        return ip, False, "video OK, audio FAIL"
    if not video_ok and audio_ok:
        return ip, False, "audio OK, video FAIL"
    return ip, False, "video FAIL, audio FAIL"


def find_active_cameras(candidate_ips: List[str], max_cameras: int) -> List[str]:
    results = []

    print("[scan] probing candidate IPs...")
    with ThreadPoolExecutor(max_workers=min(16, len(candidate_ips) or 1)) as ex:
        futures = {ex.submit(probe_camera_ip, ip): ip for ip in candidate_ips}
        for fut in as_completed(futures):
            ip, ok, msg = fut.result()
            print(f"[scan] {ip} -> {msg}")
            if ok:
                results.append(ip)
                if len(results) >= max_cameras:
                    break

    return results[:max_cameras]


# ============================================================
# Terminal launching
# ============================================================

def build_worker_command(script_name: str, xiao_ip: str, port: int) -> str:
    python_exec = sys.executable
    return (
        f'cd "{os.getcwd()}" && '
        f'"{python_exec}" "{script_name}" --xiao-ip "{xiao_ip}" --host "{HOST}" --port {port}; '
        f'echo ""; '
        f'echo "Process ended. Press Enter to close..."; '
        f'read'
    )


def launch_in_terminal(title: str, shell_command: str) -> None:
    escaped_cmd = shell_command.replace("\\", "\\\\").replace('"', '\\"')

    applescript = f'''
    tell application "Terminal"
        activate
        do script "{escaped_cmd}"
    end tell
    '''
    subprocess.Popen(["osascript", "-e", applescript])
    return
# ============================================================
# Main
# ============================================================

def main() -> None:
    # Check scripts exist
    if not Path(AUDIO_SCRIPT).exists():
        raise RuntimeError(f"Missing file: {AUDIO_SCRIPT}")
    if not Path(CAMERA_SCRIPT).exists():
        raise RuntimeError(f"Missing file: {CAMERA_SCRIPT}")

    active_ips = find_active_cameras(CANDIDATE_IPS, MAX_CAMERAS)

    if len(active_ips) < 2:
        raise RuntimeError(
            f"Needed 2 active cameras, found only {len(active_ips)}: {active_ips}"
        )

    cam1_ip, cam2_ip = active_ips[0], active_ips[1]

    print("")
    print("=== SELECTED CAMERAS ===")
    print(f"Camera 1 -> {cam1_ip}")
    print(f"  audio server  : http://127.0.0.1:{CAM1_AUDIO_PORT}/")
    print(f"  camera relay  : http://127.0.0.1:{CAM1_CAMERA_PORT}/")
    print(f"Camera 2 -> {cam2_ip}")
    print(f"  audio server  : http://127.0.0.1:{CAM2_AUDIO_PORT}/")
    print(f"  camera relay  : http://127.0.0.1:{CAM2_CAMERA_PORT}/")
    print("")

    jobs = [
        ("cam1-audio", AUDIO_SCRIPT, cam1_ip, CAM1_AUDIO_PORT),
        ("cam1-camera", CAMERA_SCRIPT, cam1_ip, CAM1_CAMERA_PORT),
        ("cam2-audio", AUDIO_SCRIPT, cam2_ip, CAM2_AUDIO_PORT),
        ("cam2-camera", CAMERA_SCRIPT, cam2_ip, CAM2_CAMERA_PORT),
    ]

    for title, script_name, ip, port in jobs:
        cmd = build_worker_command(script_name, ip, port)
        print(f"[launch] {title}: {script_name} --xiao-ip {ip} --port {port}")
        launch_in_terminal(title, cmd)

    print("")
    print("Launched 4 terminals.")
    print("Open these pages:")
    print(f"  cam1 audio  -> http://127.0.0.1:{CAM1_AUDIO_PORT}/")
    print(f"  cam1 camera -> http://127.0.0.1:{CAM1_CAMERA_PORT}/")
    print(f"  cam2 audio  -> http://127.0.0.1:{CAM2_AUDIO_PORT}/")
    print(f"  cam2 camera -> http://127.0.0.1:{CAM2_CAMERA_PORT}/")


if __name__ == "__main__":
    main()