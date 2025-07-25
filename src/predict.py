#!/usr/bin/env python3
import argparse
import importlib
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   required=True, type=str.lower)
    p.add_argument("--source",    type=lambda x: x.lower() == 'true', required=True,
                   help="True=source model, False=non-source model")
    p.add_argument("--strength",  type=int, required=True)
    return p.parse_args()

def main():
    args = parse_args()

    try:
        module = importlib.import_module(f"predict.{args.dataset}")
    except ModuleNotFoundError:
        raise SystemExit(f"[ERROR] predict/{args.dataset}.py not found.")

    if not hasattr(module, "run"):
        raise SystemExit(f"[ERROR] {args.dataset}.py lacks run(dataset, source, strength)")

    # 确保输出目录
    Path("results/predictions").mkdir(parents=True, exist_ok=True)

    module.run(args.dataset, args.source, args.strength)

if __name__ == "__main__":
    main()