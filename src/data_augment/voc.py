import argparse
import torch
import torch.nn as nn
import os
import sys
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.VOC.MSRN.models import MSRN
from models.VOC.MSRN.util import load_pretrain_model as msrn_load_pretrain_model
from dataloaders.default_voc import Voc2007Classification
from models.VOC.MSRN.engine import GCNMultiLabelMAPEngine as MSRNEngine
from .utils import DataAugmentationDataset
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
voc_info = {
    "data_name": "voc",
    "data": os.path.join("data", "source", "VOC"),
    "num_classes": 20,
    "inp_name": 'models/VOC/voc_glove_word2vec.pkl',
    "graph_file": 'models/VOC/voc_adj.pkl'
}
msrn_info = {
    "model_name": "msrn",
    "image_size": 448,
    "batch_size": 48,
    "threshold": 0.5,
    "workers": 4,
    "epochs": 100,
    "epoch_step": [30],
    "start_epoch": 0,
    "lr": 0.05,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "print_freq": 0,
    "evaluate": False,
    "pretrained": 1,
    "pretrain_model": "models/VOC_MSRN.pth.tar",
    "pool_ratio": 0.2,
    "backbone": "resnet101",
    "criterion": nn.MultiLabelSoftMarginLoss()
}
args = argparse.Namespace(**voc_info, **msrn_info)


def load_dataset(aug_method):
    trainset = Voc2007Classification(args.data, phase='trainval', inp_name=args.inp_name)
    valset = Voc2007Classification(args.data, phase='test', inp_name=args.inp_name)
    if aug_method is None or aug_method == 'no':
        return trainset, valset
    augmented_trainset = DataAugmentationDataset(
        dataname='VOC',
        dataset=trainset,
        get_img_from_item_fn=lambda item: item[0][0],
        prepare_item_fn=lambda img, item: ((img, item[0][1], item[0][2]), item[1]),
        aug_method=aug_method,
        pre_generate=False,
        save_followup=False,
    )
    return augmented_trainset, valset


def run(aug_method):
    os.makedirs('models', exist_ok=True)

    train_dataset, val_dataset = load_dataset(aug_method=aug_method)

    # train
    model = MSRN(args.num_classes, args.pool_ratio, args.backbone, args.graph_file)
    if args.pretrained:
        model = msrn_load_pretrain_model(model, args)
    model.to(device)

    criterion = args.criterion
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    state = {
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "max_epochs": args.epochs,
        "evaluate": args.evaluate,
        "num_classes": args.num_classes,
        "difficult_examples": True,
        "save_model_path": f'models/VOC/MSRN/augmented',
        "workers": args.workers,
        "epoch_step": args.epoch_step,
        "lr": args.lr,
    }
    engine = MSRNEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)
    shutil.copyfile(os.path.join(state['save_model_path'], 'model_best.pth.tar'), f'models/VOC_MSRN_Aug_{aug_method}.pth.tar')
