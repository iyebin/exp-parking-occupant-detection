"""Step 4 - Speed + bottleneck benchmark on recorded frames (offline, every config).

Same frames for every model/format/size, so the comparison is fair.
Per frame it times: preprocess, inference, postprocess (Ultralytics' own timers),
counting, and the whole predict call (total). "other" = total - (pre+inf+post+count)
is Python/framework overhead.

Between configs it waits until the Pi cools down, and it flags runs where throttling happened.

  python 04_benchmark.py                               # all 24 configs, 150 frames, 3 repeats
  python 04_benchmark.py --models yolo26n --formats ncnn --sizes 320 640
"""
import argparse
import glob
import time

import cv2
import numpy as np
import psutil

from common import (ensure_alone, FORMATS, MODELS, RECORD_DIR, RESULTS_DIR, SIZES, append_csv, arm_mhz,
                    config_id, decode_throttled, iter_configs, load_model, person_boxes, predict,
                    run_id, save_sysinfo, stats, temp_c, throttled, wait_cool)


def load_frames(patterns, n_frames, source_res):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit("no clips found - record some with 02_record.py first")
    per_clip = max(1, n_frames // len(files))
    frames = []
    for f in files:
        cap = cv2.VideoCapture(f)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(0, max(n - 1, 0), num=min(per_clip, max(n, 1)), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                if source_res:
                    fr = cv2.resize(fr, source_res, interpolation=cv2.INTER_AREA)
                frames.append(fr)
        cap.release()
    return frames[:n_frames]


def bench_one(model, fmt, frames, imgsz, conf, warmup):
    for i in range(warmup):
        predict(model, frames[i % len(frames)], fmt, imgsz, conf)
    rows = []
    for fr in frames:
        t0 = time.perf_counter()
        r = predict(model, fr, fmt, imgsz, conf)
        t1 = time.perf_counter()
        boxes = person_boxes(r, conf)
        count = int(len(boxes))
        t2 = time.perf_counter()
        sp = r.speed
        rows.append({"pre": sp.get("preprocess", np.nan), "inf": sp.get("inference", np.nan),
                     "post": sp.get("postprocess", np.nan), "count": (t2 - t1) * 1000,
                     "total": (t2 - t0) * 1000, "n": count})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=[str(RECORD_DIR / "*.avi")])
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--source-res", default="1280x720",
                    help="resize recorded frames to the live capture size ('none' = keep)")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--formats", nargs="+", default=FORMATS)
    ap.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cool-to", type=float, default=55.0)
    ap.add_argument("--threads", type=int, default=0, help="torch threads for .pt (0 = default)")
    ap.add_argument("--allow-parallel", action="store_true", help="skip the other-script check")
    a = ap.parse_args()
    ensure_alone(a.allow_parallel)

    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    rid = run_id()
    save_sysinfo(f"bench_{rid}")
    src = None if a.source_res == "none" else tuple(map(int, a.source_res.split("x")))
    frames = load_frames(a.clips, a.frames, src)
    print(f"loaded {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]})")
    proc = psutil.Process()

    for name, fmt, s in iter_configs(a.models, a.formats, a.sizes):
        cid = config_id(name, fmt, s)
        print(f"\n[{cid}]")
        try:
            model = load_model(name, fmt, s)
        except Exception as e:
            print(f"  skip: {e}")
            continue
        for rep in range(1, a.repeats + 1):
            t_start = wait_cool(a.cool_to)
            thr_before = throttled()
            psutil.cpu_percent(None)
            t0 = time.perf_counter()
            rows = bench_one(model, fmt, frames, s, a.conf, a.warmup if rep == 1 else 5)
            wall = time.perf_counter() - t0
            cpu = psutil.cpu_percent(None)
            thr_after = throttled()
            st = {k: stats([r[k] for r in rows]) for k in ["pre", "inf", "post", "count", "total"]}
            other = st["total"]["median"] - sum(st[k]["median"] for k in ["pre", "inf", "post", "count"])
            stages = {"preprocess": st["pre"]["median"], "inference": st["inf"]["median"],
                      "postprocess": st["post"]["median"], "count": st["count"]["median"],
                      "other": other}
            bottleneck = max(stages, key=stages.get)
            # throttling *during* this run = "now" bits (0-3) seen at the end, or new since-boot bits
            during = (thr_after & 0xF) if thr_after >= 0 else -1
            row = {
                "run_id": rid, "config": cid, "model": name, "format": fmt, "imgsz": s, "repeat": rep,
                "frames": len(rows), "source": f"{frames[0].shape[1]}x{frames[0].shape[0]}",
                "pre_ms_med": round(st["pre"]["median"], 2), "inf_ms_med": round(st["inf"]["median"], 2),
                "inf_ms_p95": round(st["inf"]["p95"], 2), "post_ms_med": round(st["post"]["median"], 2),
                "count_ms_med": round(st["count"]["median"], 3), "other_ms_med": round(other, 2),
                "total_ms_med": round(st["total"]["median"], 2), "total_ms_p95": round(st["total"]["p95"], 2),
                "fps": round(1000 / st["total"]["mean"], 2), "bottleneck": bottleneck,
                "bottleneck_share_pct": round(100 * stages[bottleneck] / st["total"]["median"], 1),
                "cpu_pct": round(cpu, 1), "rss_mb": round(proc.memory_info().rss / 1e6, 1),
                "temp_start_c": round(t_start, 1), "temp_end_c": round(temp_c(), 1),
                "arm_mhz_end": arm_mhz(), "throttled_during": decode_throttled(during),
                "valid": during in (0, -1),
                "mean_persons": round(float(np.mean([r["n"] for r in rows])), 2),
            }
            append_csv(RESULTS_DIR / "bench.csv", row)
            temp_txt = f"{row['temp_end_c']}C" if not np.isnan(row["temp_end_c"]) else "temp n/a"
            print(f"  rep {rep}: total {row['total_ms_med']} ms (p95 {row['total_ms_p95']}), "
                  f"inference {row['inf_ms_med']} ms, {row['fps']} fps, bottleneck={bottleneck} "
                  f"({row['bottleneck_share_pct']}%), {temp_txt}, {row['throttled_during']}")
        del model
    print(f"\nsaved {RESULTS_DIR / 'bench.csv'}")


if __name__ == "__main__":
    main()