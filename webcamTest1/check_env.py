"""Check that the Pi is ready for the camera test. Prints OK / FAIL per item.

  source ~/exitguard-venv/bin/activate
  python check_env.py              # add --device /dev/video2 if the camera is not video0

Pi CSI camera (e.g. Camera Module 3 NoIR) instead of the Kiyo Pro:
  export EXITGUARD_CAM=picam       # or: python check_env.py --device picam
"""
import argparse
import grp
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
fails = 0


def show(ok, name, detail="", warn=False):
    global fails
    tag = "OK  " if ok else ("WARN" if warn else "FAIL")
    if not ok and not warn:
        fails += 1
    print(f"[{tag}] {name:28s} {detail}")


def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def use_picam(device) -> bool:
    return str(device).lower() == "picam" or os.environ.get("EXITGUARD_CAM", "").lower() == "picam"


def check_picam():
    """Pi CSI camera (e.g. Camera Module 3 NoIR) - checked the same way the scripts open it."""
    cams = run("rpicam-hello --list-cameras 2>/dev/null | grep -E '^ *[0-9]+ :'")
    show(bool(cams), "Pi camera detected", cams.strip() or "none - check ribbon cable, then reboot")
    try:
        import picamera2  # noqa: F401
        show(True, "picamera2", "installed")
    except Exception as e:
        show(False, "picamera2", f"missing: {e.__class__.__name__} - sudo apt install -y python3-picamera2")
        return
    if not cams:
        return
    try:
        from common import PiCamCapture
        cap = PiCamCapture(1280, 720, 30)
        ok, frame = False, None
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        cap.release()
        show(ok, "camera gives frames", f"{frame.shape[1]}x{frame.shape[0]}" if ok else "read failed")
    except Exception as e:
        msg = str(e).splitlines()[0] if str(e) else e.__class__.__name__
        show(False, "camera gives frames", msg + "  (busy? stop other camera scripts / rpicam apps)")


def check_kiyo(device):
    """USB webcam (Razer Kiyo Pro)."""
    show(bool(shutil.which("v4l2-ctl")), "v4l2-ctl", "" if shutil.which("v4l2-ctl") else "sudo apt install v4l-utils")
    usb = run("lsusb | grep -i 1532")
    show(bool(usb), "Kiyo Pro on USB", usb or "not found - replug into a blue port")
    tree = run("lsusb -t")
    speed = next((ln for ln in tree.splitlines() if "uvcvideo" in ln), "")
    show("5000M" in speed or "10000M" in speed, "USB 3 port", speed.strip() or "no uvcvideo device",
         warn="480M" in speed)
    show(Path(device).exists(), f"{device} exists", "" if Path(device).exists() else run("v4l2-ctl --list-devices"))
    if Path(device).exists():
        try:
            import cv2
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = False, None
            for _ in range(10):
                ok, frame = cap.read()
                if ok:
                    break
            cap.release()
            show(ok, "camera gives frames", f"{frame.shape[1]}x{frame.shape[0]}" if ok else "read failed - replug")
        except Exception as e:
            show(False, "camera gives frames", str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0", help="/dev/videoN for the Kiyo Pro, or 'picam'")
    a = ap.parse_args()

    print("== system")
    show(platform.machine() == "aarch64", "64-bit OS", platform.machine())
    model = Path("/proc/device-tree/model").read_text(errors="ignore").strip("\x00") \
        if Path("/proc/device-tree/model").exists() else "not a Pi"
    show("Raspberry Pi 4" in model, "board", model, warn=True)
    free_gb = shutil.disk_usage("/").free / 1e9
    show(free_gb >= 3, "free disk", f"{free_gb:.1f} GB (need >= 3 GB for models + clips)")
    in_venv = sys.prefix != sys.base_prefix
    show(in_venv, "venv active", sys.prefix if in_venv else "run: source ~/exitguard-venv/bin/activate")
    show(sys.version_info >= (3, 9), "python", platform.python_version())
    try:
        in_video = "video" in [grp.getgrgid(g).gr_name for g in os.getgroups()]
    except Exception:
        in_video = False
    show(in_video, "user in 'video' group", "" if in_video else "log out/in or reboot after setup_pi.sh")

    print("== python packages")
    for mod, label in [("ultralytics", "ultralytics"), ("torch", "torch"), ("torchvision", "torchvision"),
                       ("cv2", "opencv"), ("onnx", "onnx"), ("onnxslim", "onnxslim"),
                       ("onnxruntime", "onnxruntime"), ("ncnn", "ncnn"), ("pnnx", "pnnx"),
                       ("psutil", "psutil"), ("pandas", "pandas"), ("matplotlib", "matplotlib"),
                       ("tabulate", "tabulate"), ("gpiozero", "gpiozero (LED test only)")]:
        try:
            m = importlib.import_module(mod)
            show(True, label, getattr(m, "__version__", "installed"))
        except Exception as e:
            show(False, label, f"missing: {e.__class__.__name__}", warn=(mod == "gpiozero"))
    try:
        import torch
        show(not torch.cuda.is_available() and "cu" not in torch.__version__.split("+")[-1],
             "torch is CPU build", torch.__version__)
    except Exception:
        pass
    try:
        import ultralytics
        v = tuple(int(x) for x in ultralytics.__version__.split(".")[:2])
        show(v >= (8, 4), "ultralytics >= 8.4 (YOLO26)", ultralytics.__version__)
    except Exception:
        pass
    # nvidia-ml-py is a tiny pure-Python helper that ultralytics depends on - harmless, not CUDA
    nv = run("pip list 2>/dev/null | grep -i '^nvidia' | grep -vic '^nvidia-ml-py'")
    show(nv in ("", "0"), "no CUDA packages", f"{nv} nvidia_* packages - uninstall them" if nv not in ("", "0") else "")

    if use_picam(a.device):
        print("== camera (Pi CSI camera)")
        check_picam()
    else:
        print("== camera (Kiyo Pro USB)")
        check_kiyo(a.device)

    print("== power + temperature")
    if shutil.which("vcgencmd"):
        thr = run("vcgencmd get_throttled")
        show(thr.endswith("=0x0"), "get_throttled", thr + ("" if thr.endswith("=0x0") else "  -> power supply / cooling"))
        t = run("vcgencmd measure_temp")
        try:
            tc = float(t.split("=")[1].split("'")[0])
            show(tc < 65, "idle temperature", t + ("" if tc < 65 else "  -> check fan"))
        except Exception:
            show(False, "idle temperature", t)
    else:
        show(False, "vcgencmd", "not found (not a Pi?)")
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    g = gov.read_text().strip() if gov.exists() else "n/a"
    show(g == "performance", "CPU governor", g + ("" if g == "performance" else
         "  -> echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"), warn=True)

    print("== kit")
    pts = [p for p in ("yolov8n.pt", "yolo26n.pt") if (ROOT / "models" / p).exists()]
    show(len(pts) == 2, "models exported", f"{len(pts)}/2 .pt found" + ("" if len(pts) == 2 else
         " - run 01_export_models.py (step 4)"), warn=True)

    print()
    print("READY" if fails == 0 else f"{fails} item(s) FAIL - fix them before testing (WARN = optional or later)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()