from mr_utils import mrs, paras
import argparse
import sys
import os
import random
from pathlib import Path
import torch
from PIL import Image
from itertools import permutations
from torchvision import datasets

class FollowupDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, cmr=None, transform=None):
        self.transform = transform
        self.data = []
        for idx, (img, label) in enumerate(dataset):
            if cmr is not None:
                for index in cmr:
                    img = mrs[index](img, paras[index])
            self.data.append(img)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name, e.g. MNIST')
    parser.add_argument('--strength', type=int, required=True,
                        help='Composition strength k')
    return parser.parse_args()

def validate_args(args):
    if args.dataset not in ['MNIST', 'caltech256', 'VOC', 'COCO']:
        print(f"[ERROR] Dataset '{args.dataset}' is not sopported")
        print("Supported datasets: MNIST, caltech256, VOC, COCO")
        sys.exit(1)

    if not (1 <= args.strength <= len(mrs)):
        print(f"[ERROR] Composition strength must be between 1 and {len(mrs)}")
        sys.exit(1)

def load_source(dataset: str):
    if dataset=='MNIST':
        source_inputs = datasets.MNIST(root='./data', train=False, download=False)
    elif dataset=='caltech256':
        caltech256_dataset = datasets.Caltech256(root='data', download=False)
    elif dataset=='VOC':
        pass
    else:
        
        pass
    return source_inputs

def apply_cmr(source_inputs, cmr, out_dir):
    for idx, (img, label) in enumerate(source_inputs):
        for index in cmr:
            img = mrs[index](img, paras[index])
        save_path = out_dir / f"{idx:05d}.png"
        img.save(save_path)

def main():
    args = parse_args()
    validate_args(args)
    dataset, k = args.dataset, args.strength
    source_inputs = load_source(dataset)
    for cmr in permutations(range(len(mrs)),k):
        print(cmr)
        out_dir = Path('followup') / f"{args.dataset}" / ''.join(map(str,cmr))
        out_dir.mkdir(parents=True, exist_ok=True)
        apply_cmr(source_inputs, cmr, out_dir)
    print(f"Done!")

if __name__ == '__main__':
    main()