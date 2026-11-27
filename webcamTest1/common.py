"""Shared helpers for the ExitGuard camera test (Razer Kiyo Pro + Raspberry Pi 4).

Everything here is plain Python + OpenCV so it runs on the Pi and on a laptop.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Test matrix (edit here if the team changes the plan)
# ----------------------------------------------------------------------------
MODELS = ["yolov8n", "yolo26n"]
# *_e2e = YOLO26's NMS-free head (predict/export with nms=False). Only YOLO26 has it, and
# the NCNN export cannot use it, so it exists for PyTorch and ONNX only.
FORMATS = ["pt", "onnx", "ncnn", "pt_e2e", "onnx_e2e"]
E2E_FORMATS = {"pt_e2e", "onnx_e2e"}
SIZES = [320, 416, 640, 960, 1088]  # multiples of 32; 960/1088 are slow on a Pi 4 (~1-1.5 s)
PERSON = 0  # COCO class id for "person"

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RECORD_DIR = ROOT / "recordings"


# ----------------------------------------------------------------------------
# Model files
# ----------------------------------------------------------------------------
def model_path(name: str, fmt: str, imgsz: int) -> Path:
    """Where 01_export_models.py puts each model file/folder."""
    if fmt in ("pt", "pt_e2e"):
        return MODELS_DIR / f"{name}.pt"
    if fmt == "onnx":
        return MODELS_DIR / f"{name}_{imgsz}.onnx"
    if fmt == "onnx_e2e":
        return MODELS_DIR / f"{name}_{imgsz}_e2e.onnx"
    if fmt == "ncnn":
        return MODELS_DIR / f"{name}_{imgsz}_ncnn_model"
    raise ValueError(f"unknown format {fmt}")


def predict(model, img, fmt: str, imgsz: int, conf: float, person_only: bool = True):
    """One forward pass, identical for every script. Returns the Ultralytics Results object."""
    kw = {"nms": False} if fmt in E2E_FORMATS else {}
    if person_only:
        kw["classes"] = [PERSON]
    return model.predict(img, imgsz=imgsz, conf=conf, verbose=False, **kw)[0]


def load_model(name: str, fmt: str, imgsz: int):
    from ultralytics import YOLO

    p = model_path(name, fmt, imgsz)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run 01_export_models.py first")
    return YOLO(str(p), task="detect")


def path_size_mb(p: Path) -> float:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    return p.stat().st_size / 1e6


def config_id(name: str, fmt: str, imgsz: int) -> str:
    return f"{name}_{fmt}_{imgsz}"


def iter_configs(models=None, formats=None, sizes=None):
    for m in models or MODELS:
        for f in formats or FORMATS:
            if f in E2E_FORMATS and not m.startswith("yolo26"):
                continue  # only YOLO26 has an NMS-free head
            for s in sizes or SIZES:
                yield m, f, s


# ----------------------------------------------------------------------------
# Raspberry Pi system readings (fall back to NaN on a laptop)
# ----------------------------------------------------------------------------
def _read(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def temp_c() -> float:
    v = _read("/sys/class/thermal/thermal_zone0/temp")
    return int(v) / 1000 if v else float("nan")


def arm_mhz() -> float:
    v = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    return int(v) / 1000 if v else float("nan")


def governor() -> str:
    return _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or "n/a"


def throttled() -> int:
    """vcgencmd get_throttled as an int (-1 if unavailable).

    bit 0 under-voltage now, 1 freq capped now, 2 throttled now, 3 soft temp limit now,
    bits 16-19 = the same events happened since boot.
    """
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=2).stdout
        return int(out.strip().split("=")[1], 16)
    except Exception:
        return -1


def decode_throttled(v: int) -> str:
    if v < 0:
        return "n/a"
    names = {0: "UNDER-VOLTAGE", 1: "freq-capped", 2: "THROTTLED", 3: "soft-temp-limit",
             16: "under-voltage(since boot)", 17: "freq-capped(since boot)",
             18: "throttled(since boot)", 19: "soft-temp-limit(since boot)"}
    on = [n for b, n in names.items() if v & (1 << b)]
    return ",".join(on) if on else "ok"


def wait_cool(target_c: float = 50.0, max_wait_s: int = 900) -> float:
    """Block until the SoC is at or below target_c (so every run starts from the same temperature)."""
    t0 = time.time()
    t = temp_c()
    if np.isnan(t):
        return t
    while t > target_c and time.time() - t0 < max_wait_s:
        print(f"  cooling down: {t:.1f}C > {target_c:.0f}C ...", end="\r", flush=True)
        time.sleep(5)
        t = temp_c()
    print(f"  start temperature {t:.1f}C" + " " * 20)
    return t


def system_info() -> dict:
    info = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "pi_model": (_read("/proc/device-tree/model") or "n/a").replace("\x00", ""),
        "governor": governor(),
        "kernel": platform.release(),
    }
    for mod in ["ultralytics", "torch", "cv2", "onnxruntime", "ncnn", "numpy"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "not installed"
    return info


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
def parse_res(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)

def _use_picam(device) -> bool:
    """True when the Pi CSI camera should be used: --device picam or EXITGUARD_CAM=picam."""
    import os
    return str(device).lower() == "picam" or os.environ.get("EXITGUARD_CAM", "").lower() == "picam"


class PiCamCapture:
    """Minimal cv2.VideoCapture look-alike for a CSI camera (e.g. Camera Module 3 NoIR) via Picamera2."""

    def __init__(self, width=1280, height=720, fps=30):
        from picamera2 import Picamera2
        from libcamera import controls

        self.w, self.h, self.fps = width, height, float(fps)
        self.cam = Picamera2()
        us = int(1_000_000 / fps)
        ctrls = {"FrameDurationLimits": (us, us)}
        if "AfMode" in self.cam.camera_controls:          # Camera Module 3 has autofocus
            ctrls["AfMode"] = controls.AfModeEnum.Continuous
        cfg = self.cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},  # numpy order is BGR, same as OpenCV
            controls=ctrls,
            buffer_count=2,                                       # keep latency low, no stale frames
        )
        self.cam.configure(cfg)
        self.cam.start()
        self._frame = None

    def isOpened(self):
        return True

    def grab(self):
        self._frame = self.cam.capture_array()
        return True

    def retrieve(self, *args, **kwargs):
        return self._frame is not None, self._frame

    def read(self, *args, **kwargs):
        self.grab()
        return True, self._frame

    def get(self, prop):
        import cv2
        return {cv2.CAP_PROP_FRAME_WIDTH: self.w,
                cv2.CAP_PROP_FRAME_HEIGHT: self.h,
                cv2.CAP_PROP_FPS: self.fps}.get(prop, 0.0)

    def set(self, prop, value):
        return False          # V4L2 properties don't apply to the Pi camera

    def release(self):
        self.cam.stop()
        self.cam.close()

def open_camera(device: str | int = 0, width=1280, height=720, fps=30,
                fourcc="YUYV", raw=False):
    """Open the Kiyo Pro.

    raw=True asks OpenCV for the raw YUYV buffer so colour conversion can be timed
    separately (stage 2). Use to_bgr() on each frame in that case.
    """
    import cv2

    if Path(str(device)).suffix.lower() in {".avi", ".mp4", ".mkv"}:
        # dry run with a recorded clip instead of the camera (frames arrive as BGR)
        cap = cv2.VideoCapture(str(device))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {device}")
        return cap, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     cap.get(cv2.CAP_PROP_FPS))
    if _use_picam(device):
        try:
            cap = PiCamCapture(width, height, fps)
        except Exception as e:
            raise RuntimeError(f"cannot open Pi camera: {e}\n"
                               "  busy? stop other camera scripts: ps aux | grep -E 'rpicam|python'")
        return cap, (width, height, float(fps))

    dev = int(device) if str(device).isdigit() else device
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        m = re.fullmatch(r"/dev/video(\d+)", str(dev))
        if not cap.isOpened() and m:  # some OpenCV builds only open V4L2 devices by index
            cap = cv2.VideoCapture(int(m.group(1)), cv2.CAP_V4L2)
    elif sys.platform == "win32":
        cap = cv2.VideoCapture(dev, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        hint = ""
        if sys.platform.startswith("linux"):
            import subprocess
            if not Path(str(dev)).exists() and not str(dev).isdigit():
                hint = f"\n  {dev} does not exist - camera unplugged/dropped? check: lsusb | grep 1532"
            else:
                users = subprocess.run(f"fuser {dev} 2>/dev/null", shell=True, capture_output=True,
                                       text=True).stdout.split()
                if users:
                    hint = (f"\n  camera is busy - used by process(es) {' '.join(users)}."
                            f"\n  stop the other script (Ctrl+C) or: kill {' '.join(users)}")
        raise RuntimeError(f"cannot open camera {device}{hint}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # do not queue stale frames
    if raw:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              cap.get(cv2.CAP_PROP_FPS))
    if actual[:2] != (width, height):
        print(f"WARNING: asked {width}x{height}, camera gave {actual[0]}x{actual[1]}")
    return cap, actual


def to_bgr(frame, width: int, height: int):
    """Convert a raw YUYV buffer (as returned with CONVERT_RGB=0) to BGR.

    Returns (bgr, converted_here). If OpenCV already converted, the frame is returned as-is.
    """
    import cv2

    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame, False
    if frame.size == width * height * 2:
        frame = frame.reshape(height, width, 2)
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV), True
    raise ValueError(f"unexpected raw frame shape {frame.shape}")


def set_v4l2_controls(device: str, controls: list[str]) -> None:
    """Apply v4l2-ctl controls one at a time (the Kiyo Pro can crash on some control changes)."""
    if _use_picam(device):
        print("  (Pi camera: v4l2 controls skipped)")
        return
    for c in controls:
        print(f"  v4l2-ctl --set-ctrl={c}")
        r = subprocess.run(["v4l2-ctl", "-d", str(device), f"--set-ctrl={c}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    failed: {r.stderr.strip()}")
        time.sleep(1.0)


# ----------------------------------------------------------------------------
# Detection helpers
# ----------------------------------------------------------------------------
def person_boxes(result, conf: float):
    """Return (N,5) array of [x1,y1,x2,y2,conf] for persons above conf."""
    b = result.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    xyxy = b.xyxy.cpu().numpy()
    cf = b.conf.cpu().numpy()
    cl = b.cls.cpu().numpy().astype(int)
    keep = (cl == PERSON) & (cf >= conf)
    return np.concatenate([xyxy[keep], cf[keep, None]], axis=1).astype(np.float32)


def stats(values) -> dict:
    a = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if a.size == 0:
        return {"mean": np.nan, "median": np.nan, "p95": np.nan, "max": np.nan}
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


# ----------------------------------------------------------------------------
# CSV output
# ----------------------------------------------------------------------------
def append_csv(path: Path, row: dict) -> None:
    """Append one row. If the row has columns the file doesn't (a newer script version),
    the file is rewritten once with the combined header; old rows get empty cells."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="") as fh:
            header = next(csv.reader(fh), [])
        missing = [k for k in row if k not in header]
        if missing:
            with open(path, newline="") as fh:
                old = list(csv.DictReader(fh))
            write_csv(path, old + [row])
            return
        with open(path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=header, restval="").writerow(row)
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_sysinfo(tag: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"sysinfo_{tag}.json", "w") as fh:
        json.dump(system_info(), fh, indent=2)


def run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_alone(force: bool = False) -> None:
    """Measurements are only valid if no other kit script is using the CPU or camera.
    Stops with a clear message if another 04/05/06/07/08 script (or preview) is running."""
    import os
    try:
        import psutil
    except ImportError:
        return
    me = os.getpid()
    parents = {p.pid for p in psutil.Process(me).parents()}
    names = ("04_benchmark", "05_accuracy", "06_distance", "07_live", "08_led_latency", "02_record",
             "preview.py")
    others = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] == me or p.info["pid"] in parents:
            continue
        cmd = " ".join(p.info["cmdline"] or [])
        if "python" in cmd and any(n in cmd for n in names):
            others.append(f"{p.info['pid']}: {cmd[:90]}")
    if others:
        msg = "another test script is running - results would be wrong (shared CPU/camera):\n  " + \
              "\n  ".join(others) + "\n  stop it (Ctrl+C / kill <pid>) and run again"
        if force:
            print("WARNING:", msg)
        else:
            raise SystemExit(msg)