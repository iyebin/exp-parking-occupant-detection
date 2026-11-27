"""Quick live check: does YOLO actually detect people with the current camera?

Runs each model/size for a few seconds, prints FPS + detected persons every second,
streams the annotated video to a browser, and saves the best frame per config.

  source ~/exitguard-venv/bin/activate
  export EXITGUARD_CAM=picam                  # NoIR / Pi camera (omit for the Kiyo Pro)
  python3 live_check.py                       # yolo26n + yolov8n, onnx, 640 and 320, 20 s each
  python3 live_check.py --sizes 320 --seconds 60
  python3 live_check.py --no-stream           # cleaner FPS numbers (no JPEG encoding)

Viewer: open http://localhost:8080 in your browser. VS Code Remote-SSH forwards the port
automatically (check the PORTS tab); otherwise use http://<pi-ip>:8080.
"""
from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from common import (RESULTS_DIR, ensure_alone, load_model, model_path, open_camera,
                    person_boxes, predict)

_latest = {"jpg": None}
_lock = threading.Lock()


class _MJPEG(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/stream"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        last = None
        try:
            while True:
                with _lock:
                    jpg = _latest["jpg"]
                if jpg is not None and jpg is not last:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    last = jpg
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


def start_viewer(port: int) -> None:
    srv = ThreadingHTTPServer(("0.0.0.0", port), _MJPEG)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"viewer: http://localhost:{port}  (VS Code PORTS tab, or http://<pi-ip>:{port})")


def draw(frame, boxes, label: str):
    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"person {c:.2f}", (int(x1), max(int(y1) - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame


def run_config(cap, name, fmt, sz, args, stream):
    cfg = f"{name}_{fmt}_{sz}"
    if not model_path(name, fmt, sz).exists():
        print(f"\n[{cfg}] skipped - {model_path(name, fmt, sz)} missing (run 01_export_models.py)")
        return None
    print(f"\n[{cfg}] loading ...")
    model = load_model(name, fmt, sz)
    for _ in range(3):                                   # warm-up (first runs are slow)
        ok, f = cap.read()
        if ok:
            predict(model, f, fmt, sz, args.conf)

    n = with_person = 0
    inf_ms, best, best_score = [], None, -1.0
    t0 = last = time.perf_counter()
    n_last = 0
    while time.perf_counter() - t0 < args.seconds:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        ti = time.perf_counter()
        r = predict(model, frame, fmt, sz, args.conf)
        inf_ms.append((time.perf_counter() - ti) * 1000)
        boxes = person_boxes(r, args.conf)
        n += 1
        if len(boxes):
            with_person += 1

        now = time.perf_counter()
        fps = (n - n_last) / (now - last) if now > last else 0.0
        if stream or len(boxes):
            out = draw(frame.copy(), boxes, f"{cfg}  {len(boxes)} person(s)")
            score = float(boxes[:, 4].max()) if len(boxes) else 0.0
            if score > best_score:
                best, best_score = out, score
            if stream:
                ok_j, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok_j:
                    with _lock:
                        _latest["jpg"] = jpg.tobytes()

        if now - last >= 1.0:
            confs = " ".join(f"{c:.2f}" for c in boxes[:, 4]) if len(boxes) else "-"
            print(f"  {now - t0:5.1f}s  fps {fps:5.1f}  infer {inf_ms[-1]:6.1f} ms  "
                  f"persons {len(boxes)}  conf {confs}")
            last, n_last = now, n

    elapsed = time.perf_counter() - t0
    res = {"config": cfg, "fps": n / elapsed if elapsed else 0.0,
           "infer_ms": float(np.median(inf_ms)) if inf_ms else float("nan"),
           "frames": n, "with_person": with_person}
    if best is not None:
        out_dir = RESULTS_DIR / "live_check"
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{cfg}_best.jpg"
        cv2.imwrite(str(p), best)
        res["best"] = str(p.relative_to(RESULTS_DIR.parent))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolo26n", "yolov8n"])
    ap.add_argument("--fmt", default="onnx")
    ap.add_argument("--sizes", nargs="+", type=int, default=[640, 320])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--device", default="/dev/video0", help="/dev/videoN, a video file, or 'picam'")
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-stream", action="store_true", help="no browser viewer (cleaner FPS)")
    a = ap.parse_args()

    ensure_alone()
    w, h = (int(x) for x in a.res.lower().split("x"))
    cap, actual = open_camera(a.device, w, h, 30)
    print(f"camera: {actual[0]}x{actual[1]} @ {actual[2]:.0f}")
    stream = not a.no_stream
    if stream:
        start_viewer(a.port)

    results = []
    try:
        for name in a.models:
            for sz in a.sizes:
                r = run_config(cap, name, a.fmt, sz, a, stream)
                if r:
                    results.append(r)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cap.release()

    if results:
        print("\n== summary" + ("  (FPS includes JPEG encoding for the viewer; use --no-stream for clean numbers)"
                                 if stream else ""))
        print(f"{'config':22s} {'fps':>6s} {'infer ms':>9s} {'frames w/ person':>17s}  best frame")
        for r in results:
            pct = 100 * r["with_person"] / r["frames"] if r["frames"] else 0
            print(f"{r['config']:22s} {r['fps']:6.1f} {r['infer_ms']:9.1f} "
                  f"{r['with_person']:6d}/{r['frames']:<5d} ({pct:3.0f}%)  {r.get('best', '-')}")


if __name__ == "__main__":
    main()
