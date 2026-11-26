"""Step 1 - Camera check (no model).

- Saves the camera's formats and controls (v4l2-ctl) to results/camera/.
- Measures capture-only speed for several modes: frames per second, read time,
  YUYV->BGR conversion time, failed reads. This is the ceiling for the whole pipeline.
- Optional long run (--duration 600) to check the Kiyo Pro does not drop off the USB bus.

Examples
  python 00_camera_check.py                       # 10 s per mode
  python 00_camera_check.py --modes 1280x720@60 --duration 600
  python 00_camera_check.py --set-ctrl focus_automatic_continuous=0
"""
import argparse
import subprocess
import time

from common import (RESULTS_DIR, append_csv, open_camera, run_id, save_sysinfo,
                    set_v4l2_controls, stats, to_bgr)


def v4l2_dump(device: str, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, args in {"devices": ["--list-devices"],
                       "formats": ["-d", device, "--list-formats-ext"],
                       "controls": ["-d", device, "--list-ctrls-menus"]}.items():
        try:
            r = subprocess.run(["v4l2-ctl", *args], capture_output=True, text=True, timeout=10)
            (out_dir / f"v4l2_{name}.txt").write_text(r.stdout + r.stderr)
            print(f"  saved v4l2_{name}.txt")
        except FileNotFoundError:
            print("  v4l2-ctl not found (sudo apt install v4l-utils) - skipping")
            return


def measure(device, w, h, fps, seconds, fourcc):
    cap, actual = open_camera(device, w, h, fps, fourcc=fourcc, raw=True)
    for _ in range(10):  # let auto-exposure settle
        cap.read()
    read_ms, color_ms, fails = [], [], 0
    converted = None
    t_end = time.perf_counter() + seconds
    n = 0
    t_start = time.perf_counter()
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok or frame is None:
            fails += 1
            time.sleep(0.01)
            continue
        _, converted = to_bgr(frame, actual[0], actual[1])
        t2 = time.perf_counter()
        read_ms.append((t1 - t0) * 1000)
        color_ms.append((t2 - t1) * 1000)
        n += 1
    elapsed = time.perf_counter() - t_start
    cap.release()
    r, c = stats(read_ms), stats(color_ms)
    return {
        "mode_requested": f"{w}x{h}@{fps}", "fourcc": fourcc,
        "mode_actual": f"{actual[0]}x{actual[1]}@{actual[2]:.0f}",
        "seconds": round(elapsed, 1), "frames": n, "fps_measured": round(n / elapsed, 2),
        "read_ms_median": round(r["median"], 2), "read_ms_p95": round(r["p95"], 2),
        "color_ms_median": round(c["median"], 2), "color_ms_p95": round(c["p95"], 2),
        "color_done_by": "script" if converted else "opencv",
        "failed_reads": fails,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--modes", nargs="+",
                    default=["1280x720@30", "1280x720@60", "1920x1080@30", "1920x1080@60"])
    ap.add_argument("--fourcc", default="YUYV")
    ap.add_argument("--duration", type=float, default=10, help="seconds per mode")
    ap.add_argument("--set-ctrl", nargs="*", default=[],
                    help="v4l2 controls to apply first, e.g. focus_automatic_continuous=0")
    a = ap.parse_args()

    rid = run_id()
    save_sysinfo(f"camera_{rid}")
    out_dir = RESULTS_DIR / "camera"
    print("[1] v4l2 info")
    v4l2_dump(a.device, out_dir)
    if a.set_ctrl:
        print("[2] applying controls one by one (if the camera freezes: unplug and replug)")
        set_v4l2_controls(a.device, a.set_ctrl)

    print("[3] capture-only speed")
    for m in a.modes:
        res, fps = m.split("@")
        w, h = map(int, res.split("x"))
        print(f"  {m} for {a.duration:.0f}s ...")
        try:
            row = measure(a.device, w, h, int(fps), a.duration, a.fourcc)
        except Exception as e:
            print(f"    failed: {e}")
            continue
        row = {"run_id": rid, **row}
        print(f"    -> {row['fps_measured']} fps, read {row['read_ms_median']} ms, "
              f"colour {row['color_ms_median']} ms, failed reads {row['failed_reads']}")
        append_csv(out_dir / "capture_speed.csv", row)
    print("\nAlso check the kernel log for USB errors:  dmesg | grep -iE 'uvc|usb' | tail -30")


if __name__ == "__main__":
    main()
