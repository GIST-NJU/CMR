import argparse
import torch
import numpy as np
import os
import sys
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dataloaders.default_coco import COCO2014Classification
from models.COCO.MLD.funcs import MLDecoder, mld_validate_multi
import pickle


class FollowupDataset(Dataset):
    def __init__(self, root_dir, transform=None, selected_indices=None):
        self.root_dir = root_dir
        self.transform = transform
        if selected_indices is None or len(selected_indices) == 0:
            self.images = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]
        else:
            self.images = [os.path.join(root_dir, f"{idx:05d}.png") for idx in selected_indices]
        self.cat2id = {
            'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7,
            'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13,
            'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21,
            'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28,
            'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34,
            'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40,
            'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48,
            'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56,
            'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63,
            'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70,
            'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77,
            'hair dryer': 78, 'toothbrush': 79
        }

    def __len__(self):
        return len(self.images)
    
    def get_cat2id(self):
        return self.cat2id

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return '', image, torch.zeros(args.num_classes)


class Subset(torch.utils.data.Subset):
    _local_attrs = {'dataset', 'indices'}

    def __getattr__(self, name):
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(
            f"{type(self).__name__} has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        if name in self._local_attrs:
            super().__setattr__(name, value)
        elif hasattr(self.dataset, name):
            setattr(self.dataset, name, value)
        else:
            raise AttributeError(
                f"Cannot set unknown attribute '{name}' "
                f"on {type(self).__name__}"
            )


def test_source(source_num=None):
    save_path = os.path.join(folder_path, model_name+'_source'+(f'_{source_num}' if source_num else '')+'.csv')
    if os.path.exists(save_path):
        print("Source predictions already exist.")
        return
    test_dataset = COCO2014Classification(
        args.data, args.annotation_file, phase=args.phase,
        transform=transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ])
    )
    if source_num:
        with open(f'results/samples/COCO_{source_num}.pkl', 'rb') as f:
            selected_indices = pickle.load(f)
        test_dataset = Subset(test_dataset, selected_indices)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    pred, _ = mld_validate_multi(test_loader, model, args)
    pred.to_csv(save_path, index=False)


def test_followup(augment, cmr_num=None, source_num=None):
    save_path = os.path.join(folder_path, model_name+'_followup'+(f'_{cmr_num}' if cmr_num else '')+(f'_{source_num}' if source_num else '')+'.npy')
    if os.path.exists(save_path):
        print("Followup predictions already exist.")
        return
    pred_followup = {}
    followup_dir = 'data/followup/COCO'
    if not augment:
        entries = os.listdir(followup_dir)
    else:
        with open(f'results/samples/COCO_MLD_cmr{cmr_num}.pkl', 'rb') as f:
            selected_cmrs = pickle.load(f)
        entries = set([''.join(map(str, cmr)) for cmrs in selected_cmrs.values() for cmr in cmrs])
    folders = [entry for entry in entries if os.path.isdir(os.path.join(followup_dir, entry))]
    folders = sorted(folders)
    if source_num:
        with open(f'results/samples/COCO_{source_num}.pkl', 'rb') as f:
            selected_indices = pickle.load(f)
    else:
        selected_indices = None
    for folder in tqdm(folders, desc="Predicting followup inputs"):
        cmr = tuple(int(char) for char in folder)
        followup_path = os.path.join(followup_dir, folder)
        followup_test_set = FollowupDataset(followup_path, selected_indices=selected_indices,
            transform=transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
            ]))
        followup_loader = torch.utils.data.DataLoader(followup_test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
        pred, _ = mld_validate_multi(followup_loader, model, args)
        pred_followup[cmr] = pred
    np.save(save_path, pred_followup)


def load_model(augment):
    global model, model_name, folder_path
    if not augment:
        model_name = 'COCO_MLD'
    else:
        model_name = f'COCO_MLD_Aug_{augment}'
        args.model_path = f'models/{model_name}.pth'
    model = MLDecoder(args)
    folder_path = './results/predictions/COCO'
    os.makedirs(folder_path, exist_ok=True)


def run(followup, augment, cmr_num=None, source_num=None):
    load_model(augment)
    if not followup:
        test_source(source_num)
    else:
        test_followup(augment, cmr_num, source_num)


coco_info =  {
    "data_name": "coco",
    "data": "data/source/COCO",
    "annotation_file": 'data/source/COCO/image_info_test2014.json',
    "phase": "test",
    "num_classes": 80,
}

mld_info = {
        "model_name": "mld",
        "model_type": "tresnet_l",
        "model_path": "models/COCO_MLD.pth",
        "workers": 8,
        "image_size": 448,
        "threshold": 0.5,
        "batch_size": 512,
        "print_freq": 10,
        "use_ml_decoder": 1,
        "num_of_groups": -1,
        "decoder_embedding": 768,
        "zsl": 0
    }
args = argparse.Namespace(**coco_info, **mld_info)
args.use_gpu = torch.cuda.is_available()
