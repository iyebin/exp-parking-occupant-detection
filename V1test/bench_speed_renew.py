import argparse
import csv
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from libcamera import Transform
from picamera2 import Picamera2
from ultralytics import YOLO

MODELS = {"yolov8n": 3151904, "yolo26n": 2408932}
RESOLUTIONS = [640, 960, 1088]
DURATION, WARM = 60, 10


def get_temp():
    return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000


def sample_temps(start):
    temps = []
    for sec in range(1, DURATION + 1):
        time.sleep(max(0, start + sec - time.perf_counter()))
        temps.append(get_temp())
    return temps


def make_row(second, samples, temp, duration=1):
    infer, process, people = zip(*samples) if samples else ([], [], [])
    mean = np.mean(infer) if infer else None
    proc = np.mean(process) if process else None
    return {
        "second": second,
        "frames": len(samples),
        "inference_mean_ms": round(mean, 3) if infer else "",
        "inference_p50_ms": round(np.percentile(infer, 50), 3) if infer else "",
        "inference_p95_ms": round(np.percentile(infer, 95), 3) if infer else "",
        "inference_fps": round(1000 / mean, 3) if mean else "",
        "processing_mean_ms": round(proc, 3) if process else "",
        "processing_fps": round(1000 / proc, 3) if proc else "",
        "actual_fps": round(len(samples) / duration, 3),
        "temperature_C": round(temp, 2),
        "person_counts": ";".join(map(str, people)),
        "person_mean": round(float(np.mean(people)), 3) if people else "",
        "person_min": min(people) if people else "",
        "person_max": max(people) if people else "",
    }


def bench(model, camera, res, tracking):
    predict = model.track if tracking else model.predict
    kwargs = dict(imgsz=res, rect=False, device="cpu", classes=[0],
                  conf=0.1, iou=0.7, verbose=False)
    if tracking:
        kwargs.update(tracker="bytetrack.yaml", persist=True)

    temp_start = get_temp()
    for _ in range(WARM):
        predict(camera.capture_array(), **kwargs)
    temp_warm = get_temp()

    bins = [[] for _ in range(DURATION)]
    with ThreadPoolExecutor(max_workers=1) as pool:
        start = time.perf_counter()
        future = pool.submit(sample_temps, start)
        while time.perf_counter() < start + DURATION:
            frame = camera.capture_array()
            t0 = time.perf_counter()
            result = predict(frame, **kwargs)[0]
            end = time.perf_counter()
            sec = int(end - start)
            if sec < DURATION:
                bins[sec].append((result.speed["inference"], (end - t0) * 1000,
                                  len(result.boxes)))
        temps = future.result()

    samples = [sample for bucket in bins for sample in bucket]
    rows = [make_row(i + 1, bucket, temps[i]) for i, bucket in enumerate(bins)]
    total = make_row("TOTAL", samples, np.mean(temps), DURATION)
    assert len(rows) == DURATION and sum(r["frames"] for r in rows) == total["frames"]
    rows.append(total)
    summary = {k: v for k, v in total.items() if k not in ("second", "person_counts")}
    summary.update(temp_start_C=round(temp_start, 2), temp_after_warmup_C=round(temp_warm, 2),
                   temp_mean_C=total["temperature_C"], temp_min_C=round(min(temps), 2),
                   temp_max_C=round(max(temps), 2), temp_end_C=round(temps[-1], 2),
                   temp_delta_C=round(temps[-1] - temp_start, 2))
    return rows, summary


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    for flag in ("tracking", "hflip", "vflip"):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--cooldown", type=int, default=0)
    args = parser.parse_args()
    if args.cooldown < 0:
        parser.error("--cooldown must be non-negative")

    paths = [(name, res, Path(f"onnx_models/{name}_{res}.onnx"))
             for name in MODELS for res in RESOLUTIONS]
    for _, _, path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    out = Path("results") / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    camera = Picamera2()
    summaries = []
    try:
        camera.configure(camera.create_video_configuration(
            sensor={"output_size": (1296, 972), "bit_depth": 10},
            main={"size": (1280, 960), "format": "RGB888"},
            transform=Transform(hflip=args.hflip, vflip=args.vflip),
            controls={"FrameDurationLimits": (33333, 33333)},
            queue=False,
        ))
        camera.start()
        time.sleep(2)
        print(camera.camera_configuration())
        for name, res, path in paths:
            print(f"\nTesting {name} {res}x{res}, tracking={args.tracking}", flush=True)
            model = YOLO(str(path), task="detect")
            rows, summary = bench(model, camera, res, args.tracking)
            suffix = "track" if args.tracking else "detect"
            save_csv(out / f"{name}_{res}_{suffix}.csv", rows)
            summaries.append(dict(model=name, resolution=f"{res}x{res}", device="Pi 4 CPU",
                                  tracking=args.tracking, parameters=MODELS[name],
                                  size_MB=round(path.stat().st_size / 1e6, 4),
                                  duration_s=DURATION, **summary))
            save_csv(out / "summary.csv", summaries)
            print(rows[-1], flush=True)
            del model
            gc.collect()
            if len(summaries) < len(paths):
                time.sleep(args.cooldown)
    finally:
        camera.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
