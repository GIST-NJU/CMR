import sys
import os
from mr_utils import mrs, paras
from dataloaders.default_voc import Voc2007Classification
from dataloaders.default_coco import COCO2014Classification
import argparse
from pathlib import Path
import torch
from itertools import permutations
from torchvision import datasets
from sklearn.model_selection import train_test_split
import subprocess
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import math
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name, e.g. MNIST')
    parser.add_argument('--strength', type=int, required=True,
                        help='Composition strength k')
    parser.add_argument('--num_workers', type=int, default=8)
    return parser.parse_args()

def validate_args(args):
    if args.dataset not in ['MNIST', 'Caltech256', 'VOC', 'COCO', "UTKFace"]:
        print(f"[ERROR] Dataset '{args.dataset}' is not sopported")
        print("Supported datasets: MNIST, caltech256, VOC, COCO, UTKFace")
        sys.exit(1)

    if not (1 <= args.strength <= len(mrs)):
        print(f"[ERROR] Composition strength must be between 1 and {len(mrs)}")
        sys.exit(1)

class CustomDataset(torch.utils.data.Dataset):
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

def load_source(dataset: str):
    if dataset=='MNIST':
        source_inputs = datasets.MNIST(root='./data/source', train=False, download=True)
    elif dataset=='Caltech256':
        caltech256_dataset = datasets.Caltech256(root='data/source', download=True)
        X = [caltech256_dataset[i][0] for i in range(len(caltech256_dataset))]
        y = [caltech256_dataset[i][1] for i in range(len(caltech256_dataset))]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=18, stratify=y)
        source_inputs = CustomDataset(zip(X_test, y_test))
    elif dataset=='VOC':
        source_inputs = Voc2007Classification("./data/source/VOC", phase="test", inp_name='models/VOC/voc_glove_word2vec.pkl')
    elif dataset=='COCO':
        source_inputs = COCO2014Classification("./data/source/COCO", phase="test", annotation_file='./data/source/COCO/image_info_test2014.json')
    elif dataset=='UTKFace':
        test_path = "data/source/UTKFace/UTK_test_selected"
        image_data = []
        for file in sorted(os.listdir(test_path)):
            img_path = os.path.join(test_path, file)
            try:
                img = Image.open(img_path)
                image_data.append((img, 0))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        print(f"Loaded {len(image_data)} images from UTKFace test set")
        source_inputs = CustomDataset(image_data)
    return source_inputs

def apply_cmr(dataset, source_inputs, cmr, out_dir, pbar: tqdm):
    for idx, data in enumerate(source_inputs):
        save_path = out_dir / f"{idx:05d}.png"
        if save_path.exists():
            try:
                with Image.open(save_path) as img:
                    img.verify()
                pbar.update(1)
                continue
            except Exception as e:
                print(f"{save_path} exists but cannot verify:", e)
                print(f"regenerating {save_path}")
        if dataset in ['MNIST', 'Caltech256', 'UTKFace']:
            img = data[0]
        elif dataset in ['VOC']:
            img = data[0][0]
        else:
            img = data[1]
        for index in cmr:
            img = mrs[index](img, paras[index])
        if dataset!='MNIST':
            img = img.convert('RGB')
        img.save(save_path)
        pbar.update(1)

def main():
    args = parse_args()
    validate_args(args)
    dataset, k = args.dataset, args.strength
    source_inputs = load_source(dataset)
    pbar = tqdm(desc=f"generating followup for strength {k}: ", total=math.perm(len(mrs), k) * len(source_inputs))
    with ThreadPoolExecutor(max_workers=args.num_workers) as executer:
        for cmr in permutations(range(len(mrs)), k):
            out_dir = Path('data') / Path('followup') / f"{args.dataset}" / ''.join(map(str,cmr))
            out_dir.mkdir(parents=True, exist_ok=True)
            executer.submit(apply_cmr, dataset, source_inputs, cmr, out_dir, pbar)
            time.sleep(1)
    print(f"Done!")

if __name__ == '__main__':
    main()
