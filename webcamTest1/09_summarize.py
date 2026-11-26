"""Step 9 - Merge everything into one results table + temperature charts.

Reads results/models.csv, bench.csv, accuracy.csv, distance_summary.csv, live_summary.csv
(whatever exists) and writes:
  results/final_table.csv and results/final_table.md   (one row per model/format/size)
  results/live/<run>_thermal.png                        (temperature + fps over time)

  python 09_summarize.py
  python 09_summarize.py --conf 0.25
"""
import argparse
import glob
from pathlib import Path

import pandas as pd

from common import RESULTS_DIR


def read(name):
    p = RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25, help="threshold whose accuracy goes in the table")
    a = ap.parse_args()
    key = ["model", "format", "imgsz"]

    models = read("models.csv")
    if models is None:
        raise SystemExit("results/models.csv missing - run 01_export_models.py")
    table = models[key + ["size_mb", "params", "gflops_at_imgsz"]].copy()

    bench = read("bench.csv")
    if bench is not None:
        b = bench[bench["valid"].astype(str) == "True"] if "valid" in bench else bench
        if b.empty:
            b = bench
        agg = b.groupby(key).agg(inf_ms=("inf_ms_med", "median"), total_ms=("total_ms_med", "median"),
                                 total_ms_p95=("total_ms_p95", "median"), fps=("fps", "median"),
                                 bottleneck=("bottleneck", lambda s: s.mode().iat[0]),
                                 cpu_pct=("cpu_pct", "median"), rss_mb=("rss_mb", "median")).reset_index()
        table = table.merge(agg, on=key, how="left")

    acc = read("accuracy.csv")
    if acc is not None:
        sel = acc[(acc["conf"] - a.conf).abs() < 1e-6][key + ["precision", "recall", "count_exact_rate",
                                                            "undercount_rate", "empty_false_alarm_rate"]]
        apcols = [c for c in ("ap50", "map50_95") if c in acc]
        ap50 = acc.dropna(subset=["ap50"])[key + apcols] if "ap50" in acc else None
        table = table.merge(sel, on=key, how="left")
        if ap50 is not None and not ap50.empty:
            table = table.merge(ap50, on=key, how="left")

    dist = read("distance_summary.csv")
    if dist is not None:
        col = [c for c in dist.columns if c.startswith("max_dist_m")][0]
        d = dist.pivot_table(index=["model", "imgsz"], columns="position", values=col, aggfunc="max")
        d.columns = [f"max_dist_{c}_m" for c in d.columns]
        table = table.merge(d.reset_index(), on=["model", "imgsz"], how="left")

    live = read("live_summary.csv")
    if live is not None:
        if "view" in live:  # runs with the browser view on are demos, not measurements
            live = live[live["view"].fillna("off").astype(str) != "on"]
        extra = {}
        for c in ("inf_ms_mean", "inf_ms_p50", "inf_ms_p95"):
            if c in live:
                extra[f"live_{c}"] = (c, "last")
        lv = live.sort_values("run_id").groupby(key).agg(
            **extra,
            live_fps=("fps", "last"), live_e2e_ms_p95=("e2e_ms_p95", "last"),
            live_bottleneck=("bottleneck", "last"), temp_max_c=("temp_max_c", "max"),
            first_throttle_s=("first_throttle_s", "min")).reset_index()
        table = table.merge(lv, on=key, how="left")

    table = table.sort_values(key)
    table.to_csv(RESULTS_DIR / "final_table.csv", index=False)
    try:
        (RESULTS_DIR / "final_table.md").write_text(table.to_markdown(index=False, floatfmt=".3g"))
    except ImportError:  # pandas needs 'tabulate' for markdown
        print("(pip install tabulate for final_table.md)")
    print(table.to_string(index=False))
    print(f"\nsaved {RESULTS_DIR / 'final_table.csv'} and final_table.md")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for f in glob.glob(str(RESULTS_DIR / "live" / "*_thermal.csv")):
            df = pd.read_csv(f)
            fig, ax1 = plt.subplots(figsize=(8, 3.5))
            ax1.plot(df["t_s"], df["temp_c"], color="tab:red", label="SoC temp (C)")
            ax1.axhline(80, color="tab:red", ls="--", lw=0.8)
            ax1.set_xlabel("time (s)"); ax1.set_ylabel("temp (C)")
            ax2 = ax1.twinx()
            ax2.plot(df["t_s"], df["fps"], color="tab:blue", alpha=0.7, label="fps")
            ax2.set_ylabel("fps")
            ax1.set_title(Path(f).stem.replace("_thermal", ""))
            fig.tight_layout()
            fig.savefig(f.replace(".csv", ".png"), dpi=120)
            plt.close(fig)
        print("saved thermal charts in results/live/")
    except Exception as e:
        print(f"charts skipped: {e}")


if __name__ == "__main__":
    main()