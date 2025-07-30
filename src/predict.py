#!/usr/bin/env python3
import argparse
import sys
import importlib
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name, e.g. MNIST')
    parser.add_argument('--followup', action='store_true',
                        help="Test followup inputs (default: False, test source inputs)")
    return parser.parse_args()

def validate_args(args):
    if args.dataset not in ['MNIST', 'caltech256', 'VOC', 'COCO']:
        print(f"[ERROR] Dataset '{args.dataset}' is not sopported")
        print("Supported datasets: MNIST, caltech256, VOC, COCO")
        sys.exit(1)

def main():
    args = parse_args()
    validate_args(args)
    module = importlib.import_module(f"predict.{args.dataset.lower()}")
    module.run(args.followup)

if __name__ == "__main__":
    main()