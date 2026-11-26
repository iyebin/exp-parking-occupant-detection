"""Step 7/8 - Live run on the camera: full-pipeline bottleneck, latency and heat (soak test).

A capture thread always keeps only the newest frame (no stale-frame queue). The main loop
takes it, converts YUYV->BGR, runs the model, counts people and logs every stage.

Per frame (results/live/<run>_frames.csv):
  read_ms     camera read (USB transfer + driver)          stage 1
  color_ms    YUYV -> BGR                                    stage 2
  pre/inf/post_ms  Ultralytics timers                         stages 3-5
  count_ms    counting logic                                 stage 6
  proc_ms     2..6 together (what one check costs)
  e2e_ms      from the moment the frame arrived from the camera to the count being ready
  frame_age_ms  how old the frame already was when processing started
Once per second (results/live/<run>_thermal.csv): temperature, ARM clock, throttle flags, fps, CPU%.

Safety: stops if the SoC stays >= --stop-temp (default 80C) for --stop-hold seconds,
or if under-voltage is reported.

  # latency / bottleneck, 60 s
  python 07_live.py --model yolo26n --format ncnn --imgsz 320 --duration 60
  # heat soak, 15 min, record cooling and room temperature
  python 07_live.py --model yolo26n --format ncnn --imgsz 640 --duration 900 --cooling fan --case open --ambient 22
  # watch it live from the laptop browser: http://<pi-ip>:8000  (demo only - adds CPU load)
  python 07_live.py --model yolo26n --format ncnn --imgsz 320 --duration 600 --view
"""
import argparse
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

import numpy as np
import psutil

from common import (ensure_alone, RESULTS_DIR, append_csv, arm_mhz, config_id, decode_throttled, load_model,
                    open_camera, parse_res, person_boxes, predict, run_id, save_sysinfo, stats, temp_c,
                    throttled, to_bgr, wait_cool, write_csv)


class LatestFrame:
    """Background reader that keeps only the most recent frame."""

    def __init__(self, cap, w, h, pace_fps=0.0):
        self.cap, self.w, self.h = cap, w, h
        self.pace = 1.0 / pace_fps if pace_fps else 0.0  # >0 only for a recorded clip (dry run)
        self.lock = threading.Lock()
        self.item = None  # (raw, t_arrived, read_ms, seq)
        self.seq = 0
        self.fails = 0
        self.running = True
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while self.running:
            t0 = time.perf_counter()
            ok, raw = self.cap.read()
            t1 = time.perf_counter()
            if not ok or raw is None:
                if self.pace:  # dry run: loop the clip
                    self.cap.set(1, 0)  # CAP_PROP_POS_FRAMES
                    continue
                self.fails += 1
                time.sleep(0.005)
                continue
            read_ms = (t1 - t0) * 1000  # pure read time (pacing sleep below is not counted)
            if self.pace:
                time.sleep(max(0.0, self.pace - (time.perf_counter() - t0)))
                t1 = time.perf_counter()
            with self.lock:
                self.seq += 1
                self.item = (raw, t1, read_ms, self.seq)

    def get(self, last_seq, timeout=2.0):
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            with self.lock:
                if self.item is not None and self.item[3] != last_seq:
                    return self.item
            time.sleep(0.001)
        return None

    def stop(self):
        self.running = False
        self.th.join(timeout=2)


class LiveView:
    """Tiny MJPEG server. Drawing + JPEG encoding run in their own thread at a capped rate,
    so the main loop only hands over a reference. Still costs CPU -> runs with --view are
    marked view=on in live_summary.csv and should not be used as measurement runs."""

    def __init__(self, port=8000, max_fps=5.0, width=640):
        self.port, self.period, self.width = port, 1.0 / max_fps, width
        self.lock = threading.Lock()
        self.latest = None  # (bgr, boxes, n, fps, temp)
        self.jpeg = None
        self.running = True
        view = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                    self.end_headers()
                    last = None
                    try:
                        while view.running:
                            j = view.jpeg
                            if j is not None and j is not last:
                                last = j
                                self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                                 + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                            time.sleep(0.02)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    page = (b"<html><body style='margin:0;background:#111'>"
                            b"<img src='/stream' style='width:100%;height:auto'></body></html>")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(page)

        self.srv = ThreadingHTTPServer(("0.0.0.0", port), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        threading.Thread(target=self._render, daemon=True).start()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except OSError:
            ip = "<pi-ip>"
        print(f"live view: open http://{ip}:{port} on the laptop (same network)")

    def update(self, bgr, boxes, n, fps, temp):
        with self.lock:
            self.latest = (bgr, boxes, n, fps, temp)

    def _render(self):
        while self.running:
            t0 = time.perf_counter()
            with self.lock:
                item, self.latest = self.latest, None
            if item is not None:
                bgr, boxes, n, fps, temp = item
                sc = self.width / bgr.shape[1]
                img = cv2.resize(bgr, (self.width, int(bgr.shape[0] * sc)))
                for x1, y1, x2, y2, c in boxes:
                    p1, p2 = (int(x1 * sc), int(y1 * sc)), (int(x2 * sc), int(y2 * sc))
                    cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
                    cv2.putText(img, f"{c:.2f}", (p1[0], max(12, p1[1] - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.rectangle(img, (0, 0), (self.width, 26), (0, 0, 0), -1)
                cv2.putText(img, f"people: {n}   {fps:.1f} fps   {temp:.1f}C", (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self.jpeg = buf.tobytes()
            time.sleep(max(0.0, self.period - (time.perf_counter() - t0)))

    def stop(self):
        self.running = False
        self.srv.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n")
    ap.add_argument("--format", default="ncnn")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--start-temp", type=float, default=50.0, help="cool down to this first")
    ap.add_argument("--no-cooldown", action="store_true")
    ap.add_argument("--stop-temp", type=float, default=80.0)
    ap.add_argument("--stop-hold", type=float, default=10.0)
    ap.add_argument("--cooling", default="fan", help="none / heatsink / fan (for the log)")
    ap.add_argument("--case", default="open", help="open / closed (for the log)")
    ap.add_argument("--ambient", type=float, default=float("nan"), help="room temperature C")
    ap.add_argument("--target-fps", type=float, default=0, help="limit check rate (0 = as fast as possible)")
    ap.add_argument("--view", action="store_true", help="serve annotated video at http://<pi-ip>:PORT (demo, adds load)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--view-fps", type=float, default=5.0)
    ap.add_argument("--allow-parallel", action="store_true", help="skip the other-script check")
    a = ap.parse_args()
    ensure_alone(a.allow_parallel)

    rid = run_id()
    cid = config_id(a.model, a.format, a.imgsz)
    tag = f"{rid}_{cid}"
    save_sysinfo(f"live_{tag}")
    out = RESULTS_DIR / "live"
    model = load_model(a.model, a.format, a.imgsz)
    w, h = parse_res(a.res)
    cap, actual = open_camera(a.device, w, h, a.fps, fourcc="YUYV", raw=True)
    w, h = actual[0], actual[1]
    if a.imgsz > w:
        print(f"NOTE: imgsz {a.imgsz} > camera width {w} - the frame is upscaled, no extra detail. "
              f"Use --res 1920x1080 for large input sizes.")

    # warm-up (model + camera) before the clock starts
    is_file = Path(str(a.device)).suffix.lower() in {".avi", ".mp4", ".mkv"}
    reader = LatestFrame(cap, w, h, pace_fps=(actual[2] or 30) if is_file else 0.0)
    last = 0
    for _ in range(10):
        it = reader.get(last)
        if it:
            last = it[3]
            bgr, _ = to_bgr(it[0], w, h)
            predict(model, bgr, a.format, a.imgsz, a.conf)
    t_start_c = temp_c() if a.no_cooldown else wait_cool(a.start_temp)
    viewer = LiveView(a.port, a.view_fps) if a.view else None
    cur_fps, cur_temp = 0.0, (t_start_c if t_start_c == t_start_c else 0.0)

    stop_reason = "duration"
    frames_rows = []
    hot_since = None
    first_throttle_s = None
    n_done = 0
    sec_mark = time.perf_counter()
    n_at_mark = 0
    psutil.cpu_percent(None)
    t0 = time.perf_counter()
    print(f"running {cid} for {a.duration:.0f}s at {w}x{h}@{actual[2]:.0f} (Ctrl+C to stop)")
    try:
        while time.perf_counter() - t0 < a.duration:
            it = reader.get(last)
            if it is None:
                stop_reason = "camera timeout"
                break
            raw, t_arr, read_ms, last = it
            t_s = time.perf_counter()
            bgr, _ = to_bgr(raw, w, h)
            t_c = time.perf_counter()
            r = predict(model, bgr, a.format, a.imgsz, a.conf)
            t_p = time.perf_counter()
            boxes = person_boxes(r, a.conf)
            n_person = len(boxes)
            t_e = time.perf_counter()
            if viewer:
                viewer.update(bgr, boxes, n_person, cur_fps, cur_temp)
            sp = r.speed
            frames_rows.append({
                "t_s": round(t_e - t0, 3), "frame_age_ms": round((t_s - t_arr) * 1000, 2),
                "read_ms": round(read_ms, 2), "color_ms": round((t_c - t_s) * 1000, 2),
                "pre_ms": round(sp.get("preprocess", np.nan), 2), "inf_ms": round(sp.get("inference", np.nan), 2),
                "post_ms": round(sp.get("postprocess", np.nan), 2),
                "predict_call_ms": round((t_p - t_c) * 1000, 2), "count_ms": round((t_e - t_p) * 1000, 3),
                "proc_ms": round((t_e - t_s) * 1000, 2), "e2e_ms": round((t_e - t_arr) * 1000, 2),
                "persons": n_person,
            })
            n_done += 1

            now = time.perf_counter()
            if now - sec_mark >= 1.0:
                tc, thr = temp_c(), throttled()
                fps = (n_done - n_at_mark) / (now - sec_mark)
                cur_fps, cur_temp = fps, tc
                append_csv(out / f"{tag}_thermal.csv", {
                    "t_s": round(now - t0, 1), "temp_c": tc, "arm_mhz": arm_mhz(), "throttled_hex": hex(thr) if thr >= 0 else "n/a",
                    "throttled": decode_throttled(thr), "fps": round(fps, 2), "cpu_pct": psutil.cpu_percent(None),
                    "camera_fails": reader.fails})
                print(f"  t={now - t0:6.0f}s  people={n_person}  {tc:5.1f}C  {arm_mhz():6.0f}MHz  {fps:5.2f} fps  {decode_throttled(thr):<20s}",
                      end="\r", flush=True)
                if thr >= 0 and thr & 0x6 and first_throttle_s is None:
                    first_throttle_s = round(now - t0, 1)
                if thr >= 0 and thr & 0x1:
                    stop_reason = "UNDER-VOLTAGE - use the official 5V 3A supply"
                    break
                if tc >= a.stop_temp:
                    hot_since = hot_since or now
                    if now - hot_since >= a.stop_hold:
                        stop_reason = f"safety stop: >= {a.stop_temp}C for {a.stop_hold}s"
                        break
                else:
                    hot_since = None
                sec_mark, n_at_mark = now, n_done
            if a.target_fps:
                slack = 1 / a.target_fps - (time.perf_counter() - t_s)
                if slack > 0:
                    time.sleep(slack)
    except KeyboardInterrupt:
        stop_reason = "stopped by user"
    elapsed = time.perf_counter() - t0
    reader.stop()
    cap.release()
    if viewer:
        viewer.stop()
    print()

    write_csv(out / f"{tag}_frames.csv", frames_rows)
    if not frames_rows:
        raise SystemExit(f"no frames processed ({stop_reason})")

    def med(k):
        return round(stats([r[k] for r in frames_rows])["median"], 2)

    def p95(k):
        return round(stats([r[k] for r in frames_rows])["p95"], 2)

    stages = {"read": med("read_ms"), "color": med("color_ms"), "preprocess": med("pre_ms"),
              "inference": med("inf_ms"), "postprocess": med("post_ms"), "count": med("count_ms")}
    bottleneck = max(stages, key=stages.get)
    first = [r for r in frames_rows if r["t_s"] <= 60]
    lastm = [r for r in frames_rows if r["t_s"] >= elapsed - 60]
    import csv
    temps = []
    try:
        with open(out / f"{tag}_thermal.csv") as fh:
            temps = [float(x["temp_c"]) for x in csv.DictReader(fh)]
    except Exception:
        pass
    summary = {
        "run_id": rid, "config": cid, "model": a.model, "format": a.format, "imgsz": a.imgsz,
        "camera": f"{w}x{h}@{actual[2]:.0f}", "duration_s": round(elapsed, 1), "frames": len(frames_rows),
        "fps": round(len(frames_rows) / elapsed, 2),
        "read_ms_med": stages["read"], "color_ms_med": stages["color"], "pre_ms_med": stages["preprocess"],
        "inf_ms_med": stages["inference"], "post_ms_med": stages["postprocess"], "count_ms_med": stages["count"],
        "inf_ms_mean": round(float(np.nanmean([r["inf_ms"] for r in frames_rows])), 2),
        "inf_ms_p50": med("inf_ms"), "inf_ms_p95": p95("inf_ms"),
        "proc_ms_med": med("proc_ms"), "proc_ms_p95": p95("proc_ms"),
        "e2e_ms_med": med("e2e_ms"), "e2e_ms_p95": p95("e2e_ms"), "frame_age_ms_med": med("frame_age_ms"),
        "bottleneck": bottleneck,
        "fps_first_min": round(len(first) / min(60, elapsed), 2),
        "fps_last_min": round(len(lastm) / min(60, elapsed), 2),
        "temp_start_c": round(t_start_c, 1) if not np.isnan(t_start_c) else None,
        "temp_max_c": max(temps) if temps else None,
        "first_throttle_s": first_throttle_s, "throttled_end": decode_throttled(throttled()),
        "cooling": a.cooling, "case": a.case, "ambient_c": a.ambient, "camera_fails": reader.fails,
        "view": "on" if a.view else "off",
        "stop_reason": stop_reason,
    }
    append_csv(RESULTS_DIR / "live_summary.csv", summary)
    print(f"{cid}: inference mean {summary['inf_ms_mean']} / p50 {summary['inf_ms_p50']} / "
          f"p95 {summary['inf_ms_p95']} ms")
    print(f"{cid}: {summary['fps']} fps | check {summary['proc_ms_med']} ms (p95 {summary['proc_ms_p95']}) | "
          f"latency {summary['e2e_ms_med']} ms (p95 {summary['e2e_ms_p95']}) | bottleneck {bottleneck} | "
          f"max {summary['temp_max_c']}C | {stop_reason}")


if __name__ == "__main__":
    main()