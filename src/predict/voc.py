import numpy as np
import torch
import torch.nn as nn
from itertools import permutations
import os
import argparse
import pandas as pd
import models.VOC.dataloaders.mrs as mrs
from models.VOC.dataloaders.default_voc import Voc2007Classification
from models.VOC.MSRN.models import MSRN
from models.VOC.MSRN.engine import GCNMultiLabelMAPEngine as MSRNEngine
from models.VOC.MSRN.util import load_pretrain_model as msrn_load_pretrain_model
from models.VOC.MCAR.models import MCAR
from models.VOC.MCAR.engine import MCARMultiLabelMAPEngine as MCAREngine

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

mcar_info = {
    "model_name": "mcar",
    "image_size": 448,
    "batch_size": 64,
    "threshold": 0.6,
    "bm": "resnet101",
    "ps": "avg",
    "topN": 4,
    "workers": 4,
    "epochs": 60,
    "epoch_step": [30, 50],
    "start_epoch": 0,
    "lr": 0.1,
    "lrp": 0.1,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "print_freq": 0,
    "resume": "models/VOC/MCAR/model_best_94.7850.pth.tar",
    "evaluate": True,
    "criterion": nn.BCELoss()
}

def load_model(model_name):
    if model_name == 'msrn':
        args = argparse.Namespace(**voc_info, **msrn_info)
        model = MSRN(args.num_classes, args.pool_ratio, args.backbone, args.graph_file)
        model = msrn_load_pretrain_model(model, args)
    else: # model_name == 'MCAR'
        args = argparse.Namespace(**voc_info, **mcar_info)
        model = MCAR(args)

    args.use_gpu = torch.cuda.is_available()

    return model, args

# get ouputs of source images

def test_source(model, args):
    source_test_dataset = Voc2007Classification(args.data, phase=args.phase, followup=args.followup, inp_name=args.inp_name)
    predictions = test(model, args, source_test_dataset)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    predictions.to_csv(os.path.join(args.save_path, args.data_name.upper()+'_'+args.model_name.upper()+'_source.csv'), index=False)

def test_followup(model, args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    filename = os.path.join(args.save_path, args.data_name.upper()+'_'+args.model_name.upper()+'_followup.npy')
    chunk_size = 30
    num = 0
    pred_followup = {}
    if os.path.exists(filename):
        existed = np.load(filename,allow_pickle=True).item()
    else:
        existed = {}
    for k in range(len(mrs.mrs)): 
        k = k+1
        for p in permutations(range(len(mrs.mrs)), k):
            print(p)
            if p in existed.keys():
                continue
            
            followup_test_dateset = Voc2007Classification(args.data, phase=args.phase, followup=args.followup, inp_name=args.inp_name, cmr=p)
            pred = test(model, args, followup_test_dateset)
            pred_followup[p] = pred
            
            num += 1
            if num == chunk_size:
                    pred_followup.update(existed)
                    np.save(filename, pred_followup)
                    existed = pred_followup
                    num = 0
                    pred_followup = {}

    if len(pred_followup) != 0:
        pred_followup.update(existed)
        np.save(filename, pred_followup)

def test(model, args, test_dataset):    
    engines = {
        "msrn": MSRNEngine,
        "mcar": MCAREngine,
    }
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
    if args.model_name == 'mcar':
        state.update({
            "use_pb": True
        })
    engine = engines[args.model_name](state)
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

model_names = ['msrn']
for model_name in model_names:
    print(f'----------------------------{model_name}----------------------------')
    model, args = load_model(model_name)
    args.save_path = 'predictions/VOC/'
    
    # args.followup = False
    # test_source(model, args)
   
    args.followup = True
    test_followup(model, args)