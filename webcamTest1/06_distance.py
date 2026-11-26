"""Step 6 - How far away can each config still find a person? (uses the S2 clips)

Record one person per clip at each floor mark, named  S2_d<metres>_<position>.avi
(e.g. S2_d3.0_center.avi, S2_d3.0_edge.avi). Nobody else may be in the frame.

Per clip: detection rate (frames with >=1 person / all frames), mean confidence,
median box height in pixels (source frame). Summary: the farthest distance where the
detection rate is still >= 90 % (distances below --min-dist are ignored, because a very
close person is cut off by the frame).

  python 06_distance.py
  python 06_distance.py --formats ncnn --step 2
"""
import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import (FORMATS, MODELS, RECORD_DIR, RESULTS_DIR, SIZES, config_id, iter_configs,
                    load_model, person_boxes, predict, write_csv)

PAT = re.compile(r"S2_d(?P<d>[0-9.]+)_(?P<pos>[A-Za-z]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(RECORD_DIR / "S2_d*.avi"))
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--formats", nargs="+", default=["ncnn"],
                    help="format barely changes detections; ncnn is fastest")
    ap.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--rate", type=float, default=0.9, help="detection-rate threshold")
    ap.add_argument("--min-dist", type=float, default=1.0)
    ap.add_argument("--step", type=int, default=1, help="use every n-th frame")
    a = ap.parse_args()

    clips = []
    for f in sorted(glob.glob(a.clips)):
        m = PAT.search(Path(f).stem)
        if m:
            clips.append((f, float(m["d"]), m["pos"]))
    if not clips:
        raise SystemExit("no S2_d<dist>_<pos>.avi clips found")
    print(f"{len(clips)} distance clips")

    rows, summary = [], []
    for name, fmt, s in iter_configs(a.models, a.formats, a.sizes):
        cid = config_id(name, fmt, s)
        try:
            model = load_model(name, fmt, s)
        except Exception as e:
            print(f"[{cid}] skip: {e}")
            continue
        per_pos = defaultdict(list)
        for f, dist, pos in clips:
            cap = cv2.VideoCapture(f)
            hits, confs, heights, n, i = 0, [], [], 0, 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                i += 1
                if (i - 1) % a.step:
                    continue
                r = predict(model, fr, fmt, s, a.conf)
                b = person_boxes(r, a.conf)
                n += 1
                if len(b):
                    hits += 1
                    top = b[np.argmax(b[:, 4])]
                    confs.append(float(top[4]))
                    heights.append(float(top[3] - top[1]))
            cap.release()
            rate = hits / max(n, 1)
            rows.append({"config": cid, "model": name, "format": fmt, "imgsz": s,
                         "distance_m": dist, "position": pos, "frames": n,
                         "detection_rate": round(rate, 3),
                         "mean_conf": round(float(np.mean(confs)), 3) if confs else None,
                         "median_box_h_px": round(float(np.median(heights)), 1) if heights else None})
            per_pos[pos].append((dist, rate))
            print(f"[{cid}] {dist:>4} m {pos:7s}: {rate:5.1%}")
        for pos, lst in per_pos.items():
            ok = [d for d, r in sorted(lst) if d >= a.min_dist and r >= a.rate]
            summary.append({"config": cid, "model": name, "format": fmt, "imgsz": s, "position": pos,
                            f"max_dist_m_rate{int(a.rate * 100)}": max(ok) if ok else None,
                            "tested_max_m": max(d for d, _ in lst)})
        del model
    write_csv(RESULTS_DIR / "distance.csv", rows)
    write_csv(RESULTS_DIR / "distance_summary.csv", summary)
    print(f"\nsaved {RESULTS_DIR / 'distance.csv'} and distance_summary.csv")


if __name__ == "__main__":
    main()
