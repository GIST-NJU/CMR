import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from mr_utils import mrs, paras
from dataloaders.default_voc import Voc2007Classification
from dataloaders.default_coco import COCO2014Classification
import argparse
from pathlib import Path
import torch
from PIL import Image
from itertools import permutations
from torchvision import datasets
from sklearn.model_selection import train_test_split
import subprocess


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
        tar_path = "data/caltech256/256_ObjectCategories.tar"
        extract_dir = "data/caltech256/256_ObjectCategories"  # 解压后的目录名（根据实际情况调整）
        if not Path(extract_dir).exists():
            subprocess.run(["tar", "-xf", tar_path, "-C", os.path.dirname(extract_dir)], check=True)
        caltech256_dataset = datasets.Caltech256(root='data', download=False)
        X = [caltech256_dataset[i][0] for i in range(len(caltech256_dataset))]
        y = [caltech256_dataset[i][1] for i in range(len(caltech256_dataset))]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=18, stratify=y)
        source_inputs = CustomDataset(zip(X_test, y_test))
    elif dataset=='VOC':
        source_inputs = Voc2007Classification("./data/VOC", phase="test", inp_name='models/VOC/voc_glove_word2vec.pkl')
    else:
        source_inputs = COCO2014Classification("./data/COCO", phase="test", annotation_file='./data/COCO/image_info_test2014.json')
    return source_inputs

def apply_cmr(dataset, source_inputs, cmr, out_dir):
    for idx, (img, _) in enumerate(source_inputs):
        save_path = out_dir / f"{idx:05d}.png"
        if save_path.exists():
            continue 
        if dataset in ['VOC', 'COCO']:
            img = img[0]
        for index in cmr:
            img = mrs[index](img, paras[index])
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
        apply_cmr(dataset, source_inputs, cmr, out_dir)
    print(f"Done!")

if __name__ == '__main__':
    main()