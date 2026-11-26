"""Step 5b - Detection accuracy and people counting on the labelled frames.

Needs labeled/images/*.jpg and labeled/labels/*.txt (YOLO format, class 0 = person).
For every config and every confidence threshold it reports:
  precision, recall, F1 (IoU >= 0.5), AP50, mAP50-95 (mean AP over IoU 0.50:0.05:0.95),
  counting: exact-count rate, mean absolute error, undercount rate, overcount rate,
  false-alarm rate on empty frames,
and recall per clip group (scenario / distance) in accuracy_by_group.csv.

  python 05_accuracy.py
  python 05_accuracy.py --formats ncnn --thresholds 0.15 0.25 0.35 0.5
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import (FORMATS, MODELS, RESULTS_DIR, ROOT, SIZES, config_id, iter_configs,
                    load_model, person_boxes, predict, run_id, write_csv)


def load_gt(label_path: Path, w: int, h: int) -> np.ndarray:
    if not label_path.exists():
        return None
    rows = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5 or int(float(p[0])) != 0:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


def iou(a, b):
    """IoU matrix between boxes a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(dets, gt, thr=0.5):
    """Greedy matching by confidence. Returns per-detection TP flags (sorted by conf) and #GT."""
    order = np.argsort(-dets[:, 4]) if len(dets) else np.array([], dtype=int)
    d = dets[order]
    tp = np.zeros(len(d), dtype=bool)
    used = np.zeros(len(gt), dtype=bool)
    m = iou(d[:, :4], gt)
    for i in range(len(d)):
        if len(gt) == 0:
            break
        cand = np.where(~used & (m[i] >= thr))[0]
        if len(cand):
            j = cand[np.argmax(m[i, cand])]
            used[j] = True
            tp[i] = True
    return d[:, 4], tp


def ap50(all_conf, all_tp, n_gt):
    if n_gt == 0 or len(all_conf) == 0:
        return np.nan
    order = np.argsort(-all_conf)
    tp = all_tp[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def group_of(stem: str) -> str:
    """S2_d3.0_center_f000120 -> S2_d3.0_center ; S1_empty_f000010 -> S1_empty"""
    return stem.rsplit("_f", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "labeled"))
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--formats", nargs="+", default=FORMATS)
    ap.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.15, 0.25, 0.35, 0.5])
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    img_dir, lbl_dir = Path(a.data) / "images", Path(a.data) / "labels"
    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    data = []
    for p in images:
        im = cv2.imread(str(p))
        gt = load_gt(lbl_dir / f"{p.stem}.txt", im.shape[1], im.shape[0])
        if gt is None:
            continue  # unlabeled image - skipped on purpose
        data.append((p.stem, im, gt))
    if not data:
        raise SystemExit(f"no labelled images in {a.data} (need images/ and labels/)")
    n_empty = sum(len(g) == 0 for _, _, g in data)
    print(f"{len(data)} labelled images ({n_empty} empty), {sum(len(g) for _, _, g in data)} persons")

    rid = run_id()
    rows, group_rows = [], []
    low = min(a.thresholds)
    for name, fmt, s in iter_configs(a.models, a.formats, a.sizes):
        cid = config_id(name, fmt, s)
        try:
            model = load_model(name, fmt, s)
        except Exception as e:
            print(f"[{cid}] skip: {e}")
            continue
        preds = []
        for stem, im, gt in data:
            r = predict(model, im, fmt, s, low)
            preds.append((stem, person_boxes(r, low), gt))
        for thr in a.thresholds:
            tp_all, conf_all = [], []
            n_gt = tp = fp = 0
            exact = under = over = 0
            abs_err = []
            empty_fa = 0
            groups = defaultdict(lambda: [0, 0])  # tp, n_gt
            for stem, dets, gt in preds:
                d = dets[dets[:, 4] >= thr]
                conf, tpf = match(d, gt, a.iou)
                conf_all.append(conf)
                tp_all.append(tpf)
                n_gt += len(gt)
                tp += int(tpf.sum())
                fp += int((~tpf).sum())
                diff = len(d) - len(gt)
                abs_err.append(abs(diff))
                exact += diff == 0
                under += diff < 0
                over += diff > 0
                if len(gt) == 0 and len(d) > 0:
                    empty_fa += 1
                g = groups[group_of(stem)]
                g[0] += int(tpf.sum())
                g[1] += len(gt)
            n = len(preds)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(n_gt, 1)
            # AP50 uses all detections down to the lowest threshold, computed once per config
            row = {
                "run_id": rid, "config": cid, "model": name, "format": fmt, "imgsz": s, "conf": thr,
                "images": n, "persons": n_gt, "precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
                "count_exact_rate": round(exact / n, 4), "count_mae": round(float(np.mean(abs_err)), 3),
                "undercount_rate": round(under / n, 4), "overcount_rate": round(over / n, 4),
                "empty_false_alarm_rate": round(empty_fa / max(n_empty, 1), 4) if n_empty else None,
            }
            if thr == low:
                c_all = np.concatenate(conf_all) if conf_all else np.array([])
                t_all = np.concatenate(tp_all) if tp_all else np.array([], dtype=bool)
                row["ap50"] = round(ap50(c_all, t_all, n_gt), 4)
                aps = []  # COCO-style: AP at IoU 0.50, 0.55, ..., 0.95, then the mean
                for it in np.arange(0.5, 0.951, 0.05):
                    cs, ts = [], []
                    for _, dets, gt in preds:
                        c, t = match(dets[dets[:, 4] >= thr], gt, float(it))
                        cs.append(c)
                        ts.append(t)
                    aps.append(ap50(np.concatenate(cs), np.concatenate(ts), n_gt))
                row["map50_95"] = round(float(np.nanmean(aps)), 4) if not np.all(np.isnan(aps)) else None
            else:
                row["ap50"] = None
                row["map50_95"] = None
            rows.append(row)
            for gname, (gtp, gn) in sorted(groups.items()):
                group_rows.append({"config": cid, "conf": thr, "group": gname, "persons": gn,
                                   "recall": round(gtp / gn, 4) if gn else None})
            extra = f" AP50 {row['ap50']} mAP50-95 {row['map50_95']}" if thr == low else ""
            print(f"[{cid}] conf {thr:.2f}: P {row['precision']:.3f} R {row['recall']:.3f}{extra} "
                  f"exact-count {row['count_exact_rate']:.3f} undercount {row['undercount_rate']:.3f}")
        del model
    write_csv(RESULTS_DIR / "accuracy.csv", rows)
    write_csv(RESULTS_DIR / "accuracy_by_group.csv", group_rows)
    print(f"\nsaved {RESULTS_DIR / 'accuracy.csv'} and accuracy_by_group.csv")


if __name__ == "__main__":
    main()