"""Step 5a - Sample frames from the recorded clips for labelling (ground truth).

Writes labeled/images/<clip>_f000123.jpg. Label them in CVAT / Roboflow / Label Studio with a
single class "person" (id 0) and export YOLO format into labeled/labels/ (same file names, .txt).
Images with nobody in them still need an (empty) .txt file.

  python 03_extract_frames.py --per-clip 20
  python 03_extract_frames.py --clips recordings/S1_*.avi recordings/S3_*.avi --per-clip 40
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

from common import RECORD_DIR, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=[str(RECORD_DIR / "*.avi")])
    ap.add_argument("--per-clip", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "labeled" / "images"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out.parent / "labels").mkdir(exist_ok=True)
    files = sorted({f for pat in a.clips for f in glob.glob(pat)})
    total = 0
    for f in files:
        cap = cv2.VideoCapture(f)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = np.linspace(0, max(n - 1, 0), num=min(a.per_clip, n), dtype=int)
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(str(out / f"{Path(f).stem}_f{i:06d}.jpg"), frame)
                total += 1
        cap.release()
        print(f"  {Path(f).name}: {len(idx)} frames")
    print(f"saved {total} images to {out}")
    print("next: label class 'person' (id 0), export YOLO txt files into", out.parent / "labels")


if __name__ == "__main__":
    main()
