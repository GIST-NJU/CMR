#!/usr/bin/env python3
import argparse
import sys
import importlib
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=str.lower)
    p.add_argument("--strength", type=int, required=True)
    return p.parse_args()

def validate_args(args):
    if args.dataset not in ['MNIST', 'caltech256', 'VOC', 'COCO']:
        print(f"[ERROR] Dataset '{args.dataset}' is not sopported")
        print("Supported datasets: MNIST, caltech256, VOC, COCO")
        sys.exit(1)

def main():
    args = parse_args()
    validate_args(args)
    module = importlib.import_module(f"selforacle.{args.dataset}")
    module.run(args.dataset.lower(), args.strength)

if __name__ == "__main__":
    main()