from src.mr_utils import mrs

import argparse
import sys
import os
import random
from pathlib import Path
from PIL import Image

# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name, e.g. MNIST')
    parser.add_argument('--strength', type=int, required=True,
                        help='Composition strength k')
    return parser.parse_args()

def validate_args(args):
    if args.dataset not in ['MNIST', 'Caltech256', 'VOC', 'COCO']:
        print(f"[ERROR] Dataset '{args.dataset}' is not sopported")
        sys.exit(1)

    if not (1 <= args.strength <= len(mrs)):
        print(f"[ERROR] Composition strength must be between 1 and {len(mrs)}")
        sys.exit(1)

# -----------------------------
def load_original_imgs(dataset: str):
    root = Path('data/test-inputs') / dataset
    if not root.exists():
        raise FileNotFoundError(root)
    imgs = [Image.open(p).convert('RGB') for p in root.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    return imgs

# -----------------------------
def apply_mr_chain(img, k):
    """随机选 k 个 MR 并依次作用"""
    idxs = random.sample(range(len(mu.mrs)), k)
    for i in idxs:
        img = mu.mrs[i](img, mu.paras[i])
    return img, [mu.mrs_name[i] for i in idxs]

# -----------------------------
def main():
    args = parse_args()
    validate_args(args)


    out_dir = Path('followup') / f"{args.dataset}_k{args.strength}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for j, img in enumerate(imgs):
        new_img, chain = apply_mr_chain(img, args.strength)
        new_img.save(out_dir / f"{j:05d}_{'_'.join(chain)}.png")

    print(f"Done! {len(imgs)} images → {out_dir}")

# -----------------------------
if __name__ == '__main__':
    main()
