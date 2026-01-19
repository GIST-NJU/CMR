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
    parser.add_argument('--augment', type=str, default=None, choices=[None, 'no', 'online', 'offline'],
                        help="Use which augmented model for prediction (default: None, use original model), `None`, i.e. without --augment parameter, means using the checkpoint released by model author, `no` means no augmentation but trained by us")
    parser.add_argument('--cmr_num', type=int,
                        help="Number of CMR used in augmented model evaluation")
    parser.add_argument('--source_num', type=float,
                        help="Number of source inputs used in augmented model evaluation")
    return parser.parse_args()

def validate_args(args):
    if args.dataset not in ['MNIST', 'Caltech256', 'VOC', 'COCO', 'UTKFace']:
        print(f"[ERROR] Dataset '{args.dataset}' is not sopported")
        print("Supported datasets: MNIST, caltech256, VOC, COCO, UTKFace")
        sys.exit(1)
    if args.augment is not None and args.cmr_num is None:
        print("[ERROR] Please specify --cmr_num when using augmented model for prediction")
        sys.exit(1)
    if args.augment is not None and args.source_num is None:
        print("[ERROR] Please specify --source_num when using augmented model for prediction")
        sys.exit(1)
    if not args.augment:
        args.cmr_num = None
        args.source_num = None
    if args.source_num and args.source_num >= 1:
        args.source_num = int(args.source_num)

def main():
    args = parse_args()
    validate_args(args)
    module = importlib.import_module(f"predict.{args.dataset.lower()}")
    module.run(args.followup, args.augment, args.cmr_num, args.source_num)

if __name__ == "__main__":
    main()
