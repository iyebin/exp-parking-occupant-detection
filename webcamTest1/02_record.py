"""Step 3 - Record one test clip (run once per scenario / distance mark).

Clip names follow  <scenario>_<tag>.avi  so later scripts can group results:
  S1_empty, S2_d3.0_center, S2_d3.0_edge, S3_p4_sidebyside, S4_walk_01, S5_crouch, S6_dim_p2

Writing runs in a separate thread; the script reports dropped frames.
If the Pi drops frames at 1080p, record on a laptop with the same camera settings instead.

Every clip also saves a thumbnail (middle frame) as recordings/<name>.jpg - open it in
VS Code to check the clip at a glance.

--preview shows the camera in the laptop browser (http://<pi-ip>:8000) and waits for Enter,
so you can place the person first; the view stays on (red REC) while recording. The preview
costs a little CPU - if frames_dropped > 0 with it, record without it.

Examples
  python 02_record.py --name S2_d3.0_center --seconds 5
  python 02_record.py --name S2_d3.0_center --seconds 5 --preview
  python 02_record.py --name S4_walk_01 --seconds 15 --res 1920x1080 --fps 30
"""
import argparse
import queue
import threading
import time

import cv2

from common import RECORD_DIR, append_csv, open_camera, parse_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="e.g. S2_d3.0_center")
    ap.add_argument("--seconds", type=float, default=5)
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--countdown", type=int, default=3)
    ap.add_argument("--note", default="", help="free text: lighting, people, posture ...")
    ap.add_argument("--preview", action="store_true", help="browser view + wait for Enter before recording")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--view-fps", type=float, default=4.0)
    a = ap.parse_args()

    w, h = parse_res(a.res)
    cap, actual = open_camera(a.device, w, h, a.fps, fourcc="YUYV")
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    path = RECORD_DIR / f"{a.name}.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), a.fps, (actual[0], actual[1]))
    try:
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
    except Exception:
        pass

    q: queue.Queue = queue.Queue(maxsize=120)
    dropped = 0
    stop = threading.Event()

    def write_loop():
        while not stop.is_set() or not q.empty():
            try:
                writer.write(q.get(timeout=0.1))
            except queue.Empty:
                pass

    th = threading.Thread(target=write_loop, daemon=True)
    view = None
    if a.preview:
        from viewer import WebView
        view = WebView(a.port, a.view_fps, guides=True)
        label = {"text": f"READY  {a.name}  - press Enter in the terminal to record", "col": (0, 255, 0)}
        waiting = threading.Event()
        waiting.set()

        def show_loop():  # keep the camera flowing while we wait for Enter / count down
            while waiting.is_set():
                ok, fr = cap.read()
                if ok:
                    view.update(fr, [label["text"]], label["col"])

        shower = threading.Thread(target=show_loop, daemon=True)
        shower.start()
        try:
            input(f"  {a.name}: place the person, then press Enter to record (Ctrl+C to cancel) ")
        except KeyboardInterrupt:
            waiting.clear()
            shower.join()
            cap.release()
            view.stop()
            writer.release()
            path.unlink(missing_ok=True)
            raise SystemExit("\ncancelled")
        for i in range(a.countdown, 0, -1):
            label.update(text=f"{a.name}  recording in {i} ...", col=(0, 255, 255))
            print(f"  recording in {i} ...")
            time.sleep(1)
        waiting.clear()
        shower.join()
    else:
        for i in range(a.countdown, 0, -1):
            print(f"  recording in {i} ...")
            time.sleep(1)
    for _ in range(5):
        cap.read()
    th.start()
    n = 0
    thumb = None
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a.seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        try:
            q.put_nowait(frame)
            n += 1
        except queue.Full:
            dropped += 1
        el = time.perf_counter() - t0
        if thumb is None and el >= a.seconds / 2:
            thumb = frame
        if view:
            view.update(frame, [f"REC  {a.name}  {el:4.1f}/{a.seconds:.0f}s  dropped {dropped}"], (0, 0, 255))
    elapsed = time.perf_counter() - t0
    stop.set()
    th.join()
    writer.release()
    cap.release()
    if view:
        view.stop()
    if thumb is not None:
        cv2.imwrite(str(RECORD_DIR / f"{a.name}.jpg"), thumb)

    row = {"clip": path.name, "scenario": a.name.split("_")[0], "seconds": round(elapsed, 1),
           "frames_written": n, "frames_dropped": dropped, "fps_effective": round(n / elapsed, 1),
           "resolution": f"{actual[0]}x{actual[1]}", "note": a.note}
    append_csv(RECORD_DIR / "clips.csv", row)
    print(f"saved {path}  ({n} frames, {dropped} dropped, {row['fps_effective']} fps)  thumbnail {a.name}.jpg")
    if dropped:
        print("WARNING: frames were dropped - lower the resolution or record on a laptop.")


if __name__ == "__main__":
    main()