import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import pickle
from torch.utils.data import Dataset
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.VOC.MSRN.models import MSRN
from models.VOC.MSRN.util import load_pretrain_model as msrn_load_pretrain_model
from dataloaders.default_voc import Voc2007Classification
from models.VOC.MSRN.engine import GCNMultiLabelMAPEngine as MSRNEngine

class FollowupDataset(Dataset):
    def __init__(self, root_dir, transform=None, inp_name=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return (image, '', torch.tensor(self.inp)), torch.zeros(20)

    def get_cat2id(self):
        object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'dining table', 'dog', 'horse',
                     'motorbike', 'person', 'potted plant',
                     'sheep', 'sofa', 'train', 'tv']
        self.cat2idx = {
            label: i for i, label in enumerate(object_categories)
        }
        return self.cat2idx

def test(model, args, test_dataset):
    state = {
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'max_epochs': args.epochs,
        'evaluate': True,
        'resume': args.resume,
        'num_classes': args.num_classes,
        'difficult_examples': True,
        'workers': args.workers,
        'epoch_step': args.epoch_step,
        'lr': args.lr
    }
    engine =  MSRNEngine(state)
    engine.predict(model, args.criterion, test_dataset)
    temp_cat = []
    cat_id = test_dataset.get_cat2id()
    for item in cat_id:
        temp_cat.append(item)
    _, indec = torch.sort(engine.state['ap_meter'].scores, descending=True)
    pb = torch.nn.functional.sigmoid(engine.state['ap_meter'].scores)
    result = []
    for i in range(len(indec)):
        temp = []
        temp.append(engine.state['names'][i].split(os.sep)[-1])
        for j in range(len(cat_id)):
            if pb.numpy()[i][indec.numpy()[i][j]] > args.threshold:
                temp.append(temp_cat[indec.numpy()[i][j]])
        result.append(temp)
    result = pd.DataFrame(result)
    result.rename(columns={0: "img"}, inplace=True)
    return result

def test_source():
    save_path = os.path.join(folder_path, model_name+'_source.csv')
    if os.path.exists(save_path):
        return
    predictions = test(model, args, source_test_dataset)
    predictions.to_csv(save_path, index=False)

def test_followup():
    save_path = os.path.join(folder_path, model_name+'_followup.npy')
    if os.path.exists(save_path):
        return
    pred_followup = {}
    followup_dir = 'followup/VOC'
    entries = os.listdir(followup_dir)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(followup_dir, entry))]
    folders = sorted(folders)
    for folder in folders:
        cmr = tuple(int(char) for char in folder)
        followup_path = os.path.join(followup_dir, folder)
        followup_test_set = FollowupDataset(followup_path, inp_name=args.inp_name)
        pred = test(model, args, followup_test_set)
        pred_followup[cmr] = pred
    np.save(save_path, pred_followup)

def run(followup):
    if not followup:
        test_source()
    else:
        test_followup()

voc_info = {
    "data_name": "voc",
    "data": os.path.join("data", "VOC"),
    "phase": "test",
    "num_classes": 20,
    "inp_name": 'models/VOC/voc_glove_word2vec.pkl',
    "graph_file": 'models/VOC/voc_adj.pkl'
}

msrn_info = {
    "model_name": "msrn",
    "image_size": 448,
    "batch_size": 128,
    "threshold": 0.5,
    "workers": 4,
    "epochs": 20,
    "epoch_step": [30],
    "start_epoch": 0,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "print_freq": 0,
    "resume": "models/VOC/MSRN/voc_checkpoint.pth.tar",
    "evaluate": True,
    "pretrained": 1,
    "pretrain_model": "models/VOC/MSRN/resnet101_for_msrn.pth.tar",
    "pool_ratio": 0.2,
    "backbone": "resnet101",
    "criterion": nn.MultiLabelSoftMarginLoss()
}
model_name='VOC_MSRN'
args = argparse.Namespace(**voc_info, **msrn_info)
args.use_gpu = torch.cuda.is_available()
model = MSRN(args.num_classes, args.pool_ratio, args.backbone, args.graph_file)
model = msrn_load_pretrain_model(model, args)
source_test_dataset = Voc2007Classification(args.data, phase=args.phase, inp_name=args.inp_name)
print(source_test_dataset[0][1])
folder_path = './results/predictions/VOC'
os.makedirs(folder_path, exist_ok=True)