"""Camera preview in the laptop browser - no model, very light. Use it to aim the camera
and to place people before each recording. Stop with Ctrl+C (the camera is then free
for the next script; only one program can use the camera at a time).

  python preview.py                         # 1280x720, guide lines on
  python preview.py --res 1920x1080 --fps 30
  python preview.py --no-guides --port 8001
Then open http://<pi-ip>:8000 (or http://localhost:8000 through VS Code Ports).
"""
import argparse
import time

from common import open_camera, parse_res
from viewer import WebView


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--view-fps", type=float, default=10.0)
    ap.add_argument("--no-guides", action="store_true")
    a = ap.parse_args()

    w, h = parse_res(a.res)
    cap, actual = open_camera(a.device, w, h, a.fps, fourcc="YUYV")
    view = WebView(a.port, a.view_fps, guides=not a.no_guides)
    print(f"camera {actual[0]}x{actual[1]}@{actual[2]:.0f} - Ctrl+C to stop")
    n, fails, t_mark, fps = 0, 0, time.perf_counter(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 50:
                    raise SystemExit("camera stopped giving frames - replug it")
                time.sleep(0.01)
                continue
            fails = 0
            n += 1
            now = time.perf_counter()
            if now - t_mark >= 1.0:
                fps, n, t_mark = n / (now - t_mark), 0, now
            view.update(frame, [f"PREVIEW  {actual[0]}x{actual[1]}  camera {fps:.1f} fps  "
                                f"{time.strftime('%H:%M:%S')}"])
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        view.stop()
        print("\nstopped - camera released")


if __name__ == "__main__":
    main()