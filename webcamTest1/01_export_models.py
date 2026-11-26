"""Step 2 - Download the COCO weights and export every format/size; record model size.

Creates models/<name>.pt, models/<name>_<size>.onnx, models/<name>_<size>_ncnn_model/
and results/models.csv (file size MB, parameters, GFLOPs).

Run on the Pi (or on a laptop and copy the models/ folder over).
  python 01_export_models.py
  python 01_export_models.py --models yolov8n --sizes 320
"""
import argparse
import shutil
from pathlib import Path

from common import (E2E_FORMATS, FORMATS, MODELS, MODELS_DIR, RESULTS_DIR, SIZES, model_path,
                    path_size_mb, save_sysinfo, write_csv)


def model_info(yolo, imgsz):
    """(parameters, GFLOPs) - GFLOPs at the given input size."""
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params
        return get_num_params(yolo.model), round(get_flops(yolo.model, imgsz), 2)
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--formats", nargs="+", default=FORMATS)
    ap.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    a = ap.parse_args()

    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_sysinfo("export")
    rows = []
    for name in a.models:
        pt = model_path(name, "pt", 0)
        if not pt.exists():
            print(f"[download] {name}.pt")
            m = YOLO(f"{name}.pt")  # downloads into the current dir
            src = Path(m.ckpt_path) if getattr(m, "ckpt_path", None) else Path(f"{name}.pt")
            shutil.copy(src, pt)
        base = YOLO(str(pt))
        params = model_info(base, 640)[0]
        e2e = getattr(base.model, "end2end", None)
        print(f"[{name}] params={params}  NMS-free head on by default: {e2e}")

        for s in a.sizes:
            flops = model_info(base, s)[1]
            for fmt in a.formats:
                if fmt in E2E_FORMATS and not name.startswith("yolo26"):
                    continue
                dst = model_path(name, fmt, s)
                if fmt not in ("pt", "pt_e2e") and not dst.exists():
                    print(f"[export] {name} {fmt} {s}")
                    try:
                        m = YOLO(str(pt))
                        kw = {"nms": False} if fmt in E2E_FORMATS else {}
                        out = Path(m.export(format=fmt.replace("_e2e", ""), imgsz=s, half=False,
                                            dynamic=False, simplify=True, verbose=False, **kw))
                        shutil.move(str(out), str(dst))
                    except Exception as e:
                        print(f"  EXPORT FAILED ({name} {fmt} {s}): {e}")
                        continue
                rows.append({
                    "model": name, "format": fmt, "imgsz": s,
                    "path": str(dst.relative_to(MODELS_DIR.parent)),
                    "size_mb": round(path_size_mb(dst), 2),
                    "params": params, "gflops_at_imgsz": flops,
                    "precision": "fp32", "nms_free_head": fmt in E2E_FORMATS,
                })
    # merge with earlier runs (e.g. 320 exported today, 640 tomorrow)
    out = RESULTS_DIR / "models.csv"
    if out.exists():
        import csv
        with open(out) as fh:
            old = [r for r in csv.DictReader(fh)]
        new_keys = {(r["model"], r["format"], str(r["imgsz"])) for r in rows}
        rows = [r for r in old if (r["model"], r["format"], r["imgsz"]) not in new_keys] + rows
    write_csv(out, rows)
    print(f"\nsaved {RESULTS_DIR / 'models.csv'}")
    for r in rows:
        print(f"  {r['model']:8s} {r['format']:9s} {int(r['imgsz']):4d}  {float(r['size_mb']):7.2f} MB")


if __name__ == "__main__":
    main()