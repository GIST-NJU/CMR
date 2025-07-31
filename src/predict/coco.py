import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import pickle
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dataloaders.default_coco import COCO2014Classification
from models.COCO.MLD.funcs import MLDecoder, mld_validate_multi

class FollowupDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
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


def test_source():
    save_path = os.path.join(folder_path, model_name+'_source.csv')
    if os.path.exists(save_path):
        return
    test_dataset = COCO2014Classification(
            args.data, args.annotation_file, phase=args.phase,
            transform=transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
            ])
        )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    pred, map = mld_validate_multi(test_loader, model, args)
    pred.to_csv(save_path, index=False)

def test_followup():
    save_path = os.path.join(folder_path, model_name+'_followup.npy')
    if os.path.exists(save_path):
        return
    pred_followup = {}
    followup_dir = 'followup/COCO'
    entries = os.listdir(followup_dir)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(followup_dir, entry))]
    folders = sorted(folders)
    for folder in folders:
        cmr = tuple(int(char) for char in folder)
        followup_path = os.path.join(followup_dir, folder)
        followup_test_set = FollowupDataset(followup_path,
            transform=transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
            ]))
        followup_loader = torch.utils.data.DataLoader(followup_test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
        pred, map = mld_validate_multi(followup_loader, model, args)
        pred_followup[cmr] = pred
    np.save(save_path, pred_followup)

def run(followup):
    if not followup:
        test_source()
    else:
        test_followup()

coco_info =  {
    "data_name": "coco",
    "data": "data/COCO",
    "annotation_file": 'data/COCO/image_info_test2014.json',
    "phase": "test",
    "num_classes": 80,
}

mld_info = {
        "model_name": "mld",
        "model_type": "tresnet_l",
        "model_path": "models/COCO/MLD/tresnet_l_COCO__448_90_0.pth",
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

model_name='COCO_MLD'
args = argparse.Namespace(**coco_info, **mld_info)
args.use_gpu = torch.cuda.is_available()
model = MLDecoder(args)

folder_path = './results/predictions/COCO'
os.makedirs(folder_path, exist_ok=True)