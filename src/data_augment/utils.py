import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from mr_utils import mrs, paras

import random
from tqdm import tqdm
from torch.utils.data import Dataset
import torch


class DefaultDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.transform = transform
        self.data = list(zip(X, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class DataAugmentationDataset(Dataset):
    def __init__(
            self,
            dataname,
            dataset: Dataset,
            get_img_from_item_fn,
            prepare_item_fn,
            aug_method='offline',  # or 'online',
            pre_generate=True,
            save_followup=True,
            transform=None,
        ):
        self.dataname = dataname
        self.dataset = dataset
        # use to get image from dataset[idx], just like lambda item: item[0]
        # e.g., for mnist, dataset[idx] returns (img, label), we get item[0]
        self.get_img_from_item_fn = get_img_from_item_fn
        # "inverse function" of get_img_from_item_fn
        self.prepare_item_fn = prepare_item_fn
        self.mrs = mrs
        self.paras = paras
        self.transform = transform
        # offline: `pre-generate` all augmented data (use all source + followup images as training data)
        # online: `on-the-fly` augment data during training (randomly apply one CMR to each image in each epoch, the size of dataset remains the same as source dataset)
        self.aug_method = aug_method
        self.pre_generate = pre_generate  # only for offline
        self.save_followup = save_followup  # only for offline and pre_generate
        if self.aug_method == 'offline' and self.pre_generate:
            self.augmented_data = self.generate_augmented_data()
        print(f"[DataAugmentationDataset] dataname={dataname}, aug_method={aug_method}, size={len(self)}")

    def __len__(self):
        if self.aug_method == 'offline':
            return len(self.dataset) * (len(self.mrs) + 1)
        else:
            return len(self.dataset)

    def generate_augmented_data(self):
        augmented_data_path = os.path.join('data', 'augmented_data', f"{self.dataname}.pt")
        if os.path.exists(augmented_data_path):
            print(f"Loading augmented data from {augmented_data_path}")
            return torch.load(augmented_data_path)
        augmented_data = []
        for idx in tqdm(range(len(self.dataset)), desc="Generating augmented data"):
            item = self.dataset[idx]
            img = self.get_img_from_item_fn(item)
            for mr_idx in range(len(self.mrs)):
                aug_img = self.mrs[mr_idx](img, self.paras[mr_idx])
                augmented_data.append(self.prepare_item_fn(aug_img, item))
        os.makedirs(os.path.dirname(augmented_data_path), exist_ok=True)
        if self.save_followup:
            torch.save(augmented_data, augmented_data_path)
        return augmented_data

    def __getitem__(self, idx):
        if self.aug_method == 'offline':
            if idx < len(self.dataset):  # source image
                item = self.dataset[idx]
                if self.transform is not None:
                    img = self.get_img_from_item_fn(item)
                    img = self.transform(img)
                    item = self.prepare_item_fn(img, item)
                return item
            else:  # augmented image
                aug_idx = idx - len(self.dataset)
                if self.pre_generate:
                    item = self.augmented_data[aug_idx]
                    if self.transform is not None:
                        img = self.get_img_from_item_fn(item)
                        img = self.transform(img)
                        item = self.prepare_item_fn(img, item)
                else:
                    mr_idx = aug_idx % len(self.mrs)
                    img_idx = aug_idx // len(self.mrs)
                    item = self.dataset[img_idx]
                    img = self.get_img_from_item_fn(item)
                    aug_img = self.mrs[mr_idx](img, self.paras[mr_idx])
                    if self.transform is not None:
                        aug_img = self.transform(aug_img)
                    item = self.prepare_item_fn(aug_img, item)
                return item
        else:  # online
            item = self.dataset[idx]
            img = self.get_img_from_item_fn(item)
            mr_idx = random.randint(0, len(self.mrs))
            if mr_idx < len(self.mrs):
                img = self.mrs[mr_idx](img, self.paras[mr_idx])
            if self.transform is not None:
                img = self.transform(img)
            return self.prepare_item_fn(img, item)
