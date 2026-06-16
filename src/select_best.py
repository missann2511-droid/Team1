"""Select the best trained model and publish it as the production artifact.

Reads every <model>_results.json, picks the lowest RMSE, copies the matching
fitted pipeline to models/best/model_pipeline.joblib, and writes a combined
model_comparison.json. Run after all parallel training jobs finish.
"""
from __future__ import annotations

import json
import shutil

import joblib

from .config import BEST_DIR, MODELS_DIR, MODEL_NAMES, PROCESSED_DIR


def run() -> dict:
    results = []
    for name in MODEL_NAMES:
        p = PROCESSED_DIR / f"{name}_results.json"
        if p.exists():
            results.append(json.loads(p.read_text()))
        else:
            print(f"[select] WARNING: missing results for {name}")

    if not results:
        raise SystemExit("[select] No results found — run training first.")

    results.sort(key=lambda r: r["rmse"])
    best = results[0]
    print("[select] Leaderboard (by RMSE):")
    for r in results:
        star = " <-- BEST" if r is best else ""
        print(f"    {r['model']:18s} RMSE=${r['rmse']:>9,.2f}  R2={r['r2']:.4f}{star}")

    src = MODELS_DIR / f"{best['model']}_pipeline.joblib"
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    dst = BEST_DIR / "model_pipeline.joblib"
    shutil.copyfile(src, dst)
    # sanity load
    joblib.load(dst)

    comparison = {"best_model": best["model"], "results": results}
    (PROCESSED_DIR / "model_comparison.json").write_text(json.dumps(comparison, indent=2))
    (BEST_DIR / "metadata.json").write_text(json.dumps(best, indent=2))
    print(f"[select] Published {best['model']} -> {dst}")
    return comparison


if __name__ == "__main__":
    run()
