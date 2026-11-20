"""Maia-2 predictions on a human test set (chess-rl human testset), for `chess-rl human eval --preds`.

Maia-2 (Tang et al., NeurIPS 2024; MIT licence) is not a project dependency: run this with a separate
environment that has the `maia2` package, e.g.

    uv venv /tmp/maia && VIRTUAL_ENV=/tmp/maia uv pip install maia2
    /tmp/maia/bin/python scripts/maia2_baseline.py data/lichess/human-test-2026-09.jsonl runs/human-eval/maia2.jsonl

Blitz rows use Maia-2's blitz model; rapid and classical rows use its rapid model. Output, one row per
position: {"id", "top1", "p_move"}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd
from maia2 import inference, model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_set")
    parser.add_argument("out")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--models-dir", default="runs/maia2_models")
    args = parser.parse_args()

    with open(args.test_set) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    df["board"], df["active_elo"], df["opponent_elo"] = df["fen"], df["elo_us"], df["elo_them"]
    df["maia"] = df["time_class"].map(lambda tc: "blitz" if tc in ("blitz", "bullet") else "rapid")

    out = []
    for kind, part in df.groupby("maia"):
        t0 = time.time()
        net = model.from_pretrained(type=kind, device=args.device, save_root=args.models_dir)
        part = part.reset_index(drop=True)
        scored, acc = inference.inference_batch(part.copy(), net, False, args.batch_size, 4)
        for move, probs, rid in zip(scored["move"], scored["move_probs"], scored["id"], strict=True):
            out.append({"id": rid, "top1": next(iter(probs)), "p_move": probs.get(move, 0.0)})
        print(f"maia2 {kind}: {len(part):,} positions, top-1 {acc:.4f}, {time.time() - t0:.0f}s", file=sys.stderr)

    with open(args.out, "w") as f:
        for item in out:
            f.write(json.dumps(item) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
