#!/usr/bin/env python3
import argparse
import sys
import importlib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name, e.g. MNIST')
    parser.add_argument('--augment', type=str, choices=['no', 'online', 'offline'], default='online',
                        help='Augmentation method, e.g. online or offline')
    return parser.parse_args()

def validate_args(args):
    if args.dataset not in ['MNIST', 'Caltech256', 'VOC', 'COCO', 'UTKFace']:
        print(f"[ERROR] Dataset '{args.dataset}' is not sopported")
        print("Supported datasets: MNIST, caltech256, VOC, COCO, UTKFace")
        sys.exit(1)

def main():
    args = parse_args()
    validate_args(args)
    module = importlib.import_module(f"data_augment.{args.dataset.lower()}")
    module.run(args.augment)

if __name__ == "__main__":
    main()
