"""Step 8 (optional) - Real camera latency with an LED ("glass-to-result").

Software timing (07_live.py) starts when the frame reaches Python, so it misses the time the
camera and USB need to deliver it. Here the Pi switches an LED on at a known moment and we
wait for the first frame in which the LED looks bright.

  camera_latency  = LED on -> first bright frame arrives in Python
  + processing    = that frame through the model (from 07_live.py, or measured here with --model)

Setup: point the camera at the LED (traffic-light module KS0310, red pin on GPIO17 by default),
dim the room a little, and give the LED area with --roi x,y,w,h (pixels). Use --pick-roi to
print the brightest-change region automatically.

  python 08_led_latency.py --pick-roi
  python 08_led_latency.py --roi 600,320,40,40 --trials 30
  python 08_led_latency.py --roi 600,320,40,40 --trials 30 --model yolo26n --format ncnn --imgsz 320
"""
import argparse
import time

import numpy as np

from common import (RESULTS_DIR, append_csv, load_model, open_camera, parse_res, predict, run_id, stats,
                    to_bgr)


def roi_mean(bgr, roi):
    x, y, w, h = roi
    return float(bgr[y:y + h, x:x + w].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpio", type=int, default=17)
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--roi", default=None, help="x,y,w,h of the LED in the image")
    ap.add_argument("--pick-roi", action="store_true")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--delta", type=float, default=25.0, help="brightness jump that counts as 'on'")
    ap.add_argument("--model", default=None)
    ap.add_argument("--format", default="ncnn")
    ap.add_argument("--imgsz", type=int, default=320)
    a = ap.parse_args()

    from gpiozero import LED  # installed with Raspberry Pi OS

    led = LED(a.gpio)
    w, h = parse_res(a.res)
    cap, actual = open_camera(a.device, w, h, a.fps, fourcc="YUYV", raw=True)
    w, h = actual[0], actual[1]

    def frame():
        ok, raw = cap.read()
        return to_bgr(raw, w, h)[0] if ok else None

    for _ in range(20):
        cap.read()

    if a.pick_roi:
        led.off(); time.sleep(0.5)
        off = np.mean([frame().astype(np.float32) for _ in range(5)], axis=0)
        led.on(); time.sleep(0.5)
        on = np.mean([frame().astype(np.float32) for _ in range(5)], axis=0)
        led.off()
        diff = (on - off).mean(axis=2)
        y, x = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"brightest change at x={x}, y={y}  ->  try --roi {max(x - 20, 0)},{max(y - 20, 0)},40,40")
        return

    roi = tuple(map(int, a.roi.split(",")))
    model = load_model(a.model, a.format, a.imgsz) if a.model else None
    cam_ms, proc_ms = [], []
    for i in range(a.trials):
        led.off()
        time.sleep(0.6)
        for _ in range(3):  # flush
            cap.read()
        base = np.median([roi_mean(frame(), roi) for _ in range(5)])
        t_on = time.perf_counter()
        led.on()
        got = None
        while time.perf_counter() - t_on < 1.0:
            fr = frame()
            if fr is not None and roi_mean(fr, roi) > base + a.delta:
                got = time.perf_counter()
                break
        if got is None:
            print(f"  trial {i + 1}: LED not seen - check --roi / --delta")
            continue
        cam_ms.append((got - t_on) * 1000)
        if model:
            t0 = time.perf_counter()
            predict(model, fr, a.format, a.imgsz, 0.25)
            proc_ms.append((time.perf_counter() - t0) * 1000)
        print(f"  trial {i + 1}: camera {cam_ms[-1]:.1f} ms" + (f", +model {proc_ms[-1]:.1f} ms" if model else ""))
    led.off()
    cap.release()
    c = stats(cam_ms)
    row = {"run_id": run_id(), "camera": f"{w}x{h}@{actual[2]:.0f}", "trials": len(cam_ms),
           "camera_latency_ms_med": round(c["median"], 1), "camera_latency_ms_p95": round(c["p95"], 1),
           "frame_interval_ms": round(1000 / actual[2], 1) if actual[2] else None}
    if model:
        p = stats(proc_ms)
        row.update({"config": f"{a.model}_{a.format}_{a.imgsz}", "proc_ms_med": round(p["median"], 1),
                    "glass_to_result_ms_med": round(c["median"] + p["median"], 1)})
    append_csv(RESULTS_DIR / "led_latency.csv", row)
    print(row)
    print("note: the result is only accurate to about one frame interval.")


if __name__ == "__main__":
    main()
