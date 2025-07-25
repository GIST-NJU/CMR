import numpy as np
import torch
import torch.nn as nn
from itertools import permutations
import os
import argparse
import pandas as pd
import dataloaders.mrs as mrs
import torchvision.transforms as transforms

# load model
from dataloaders.default_coco import COCO2014Classification
from models.COCO.MLD.funcs import MLDecoder, mld_validate_multi
from models.COCO.ASL.funcs import ASL, asl_validate_multi

coco_info =  {
    "data_name": "coco",
    "data": "/bbfs/scratch/hyw_husy/COCO",
    "annotation_file":"/bbfs/scratch/hyw_husy/COCO/image_info_test2014_followup_tmp.json",
    "phase": "test",
    "num_classes": 80,
}

mld_info = {
        "model_name": "mld",
        "model_type": "tresnet_l",
        "model_path": "models/COCO/MLD/tresnet_l_COCO__448_90_0.pth",
        "workers": 4,
        "image_size": 448,
        "threshold": 0.5,
        "batch_size": 1,
        "print_freq": 10,
        "use_ml_decoder": 1,
        "num_of_groups": -1,
        "decoder_embedding": 768,
        "zsl": 0
    }

asl_info = {
        "model_name": "asl",
        "model_type": "tresnet_l",
        "model_path": "models/COCO/ASL/MS_COCO_TRresNet_L_448_86.6.pth",
        "workers": 4,
        "image_size": 448,
        "threshold": 0.8,
        "batch_size": 1,
        "print_freq": 10,
    }

def load_model(model_name):
    if model_name == 'mld':
        args = argparse.Namespace(**coco_info, **mld_info)
        model = MLDecoder(args)
    else: # model_name == 'ASL'
        args = argparse.Namespace(**coco_info, **asl_info)
        model = ASL(args)
        pass

    args.use_gpu = torch.cuda.is_available()

    return model, args

# get ouputs of source images
def test_source(model, args):
    pred = test(model, args, COCO2014Classification)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    print(os.path.join(args.save_path, args.data_name.upper()+'_'+args.model_name.upper()+'_source.csv'))
    pred.to_csv(os.path.join(args.save_path, args.data_name.upper()+'_'+args.model_name.upper()+'_source.csv'), index=False)

# get outputs of followup images
def test_followup(model, args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    filename = os.path.join(args.save_path, args.data_name.upper()+'_'+args.model_name.upper()+'_followup.npy')
    chunk_size = 100
    num = 0
    pred_followup = {}
    if os.path.exists(filename):
        existed = np.load(filename,allow_pickle=True).item()
    else:
        existed = {}
    for k in range(1): 
        k = k+1
        for p in permutations(range(len(mrs.mrs)), k):
            print(p)
            if p in existed.keys():
                continue
            args.cmr = p
            pred = test(model, args, COCO2014Classification)
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

def test(model, args, dataloader_class):    
    if args.model_name == 'mld':
        val_dataset = dataloader_class(
            args.data, args.annotation_file, phase=args.phase,
            followup=args.followup, cmr = args.cmr,
            transform=transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
            ])
        )
    elif args.model_name == 'asl':
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        val_dataset = dataloader_class(
            args.data, args.annotation_file, phase=args.phase,
            followup=args.followup, cmr = args.cmr,
            transform=transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        raise NotImplementedError("model {} is not implemented".format(args.model_name))
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.model_name == 'mld':
        pred, map = mld_validate_multi(val_loader, model, args)
    else: # args.model_name == 'asl'
        pred, map = asl_validate_multi(val_loader, model, args)
    print(len(pred))
    return pred


model_names = ['mld', 'asl']
print(os.getcwd())
for model_name in model_names:
    print(f'----------------------------{model_name}----------------------------')
    model, args = load_model(model_name)
    args.save_path = 'predictions/COCO/'
    args.followup = False
    args.cmr = None
    test_source(model, args)
    # args.followup = True
    # test_followup(model, args)