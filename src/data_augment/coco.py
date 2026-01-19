import argparse
import torch
import os
import sys
import shutil
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.COCO.MLD.helper_functions.helper_functions  import mAP, CocoDetection, CutoutPIL, ModelEma, \
    add_weight_decay
from models.COCO.MLD.models import create_model
from models.COCO.MLD.loss_functions.losses import AsymmetricLoss

import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast

from .utils import DataAugmentationDataset
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coco_info =  {
    "data_name": "coco",
    "data": "data/source/COCO",
    "num_classes": 80,
}
mld_info = {
    "model_name": "mld",
    "model_type": "tresnet_l",
    "model_path": "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth",
    "workers": 8,
    "image_size": 448,
    "threshold": 0.5,
    "lr": 1e-4,
    "batch_size": 96,
    "use_ml_decoder": 1,
    "num_of_groups": -1,
    "decoder_embedding": 768,
    "zsl": 0
}
args = argparse.Namespace(**coco_info, **mld_info)


def load_dataset(aug_method):
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        CutoutPIL(cutout_factor=0.5),
        RandAugment(),
        transforms.ToTensor(),
    ])
    trainset = CocoDetection(
        f"{args.data}/train2014",
        annFile="data/source/COCO/instances_train2014.json"
    )
    valset = CocoDetection(
        f"{args.data}/val2014",
        annFile="data/source/COCO/instances_val2014.json",
        transform=test_transform
    )
    if aug_method is None or aug_method == 'no':
        trainset.transform = train_transform
        return trainset, valset
    augmented_trainset = DataAugmentationDataset(
        dataname='COCO',
        dataset=trainset,
        get_img_from_item_fn=lambda item: item[0],
        prepare_item_fn=lambda img, item: (img, item[1]),
        aug_method=aug_method,
        transform=train_transform,
        pre_generate=False,
        save_followup=False,
    )
    return augmented_trainset, valset


def train_multi_label_coco(model, train_loader, val_loader, lr):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 40
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    scaler = GradScaler()
    for epoch in range(Epochs):
        for i, (inputData, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{Epochs}')):
            inputData = inputData.cuda()
            target = target.cuda()
            target = target.max(dim=1)[0]
            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !
            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            ema.update(model)
        # store information
        print('Epoch [{}/{}], LR {:.1e}, Loss: {:.1f}'
              .format(epoch, Epochs, scheduler.get_last_lr()[0],
                      loss.item()))

        try:
            os.makedirs(os.path.join('models', 'COCO', 'MLD', 'augmented'), exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_classes": args.num_classes
                },
                os.path.join('models', 'COCO', 'MLD', 'augmented', 'checkpoint.pth')
            )
        except:
            pass

        model.eval()

        mAP_score = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            shutil.copy(
                os.path.join('models', 'COCO', 'MLD', 'augmented', 'checkpoint.pth'),
                os.path.join('models', 'COCO', 'MLD', 'augmented', 'model_best.pth')
            )
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))
    shutil.copy(
        os.path.join('models', 'COCO', 'MLD', 'augmented', 'checkpoint.pth'),
        os.path.join('models', 'COCO', 'MLD', 'augmented', f'model_best_{highest_mAP:.4f}.pth')
    )

def validate_multi(val_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(tqdm(val_loader, desc='Validating')):
        target = target
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)


def run(aug_method):
    print('creating model {}...'.format(args.model_name))
    model = create_model(args).cuda()
    print('done')

    train_dataset, val_dataset = load_dataset(aug_method=aug_method)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False
    )
    train_multi_label_coco(model, train_loader, val_loader, args.lr)
    shutil.copyfile(
        os.path.join('models', 'COCO', 'MLD', 'augmented', 'model_best.pth'),
        f'models/COCO_MLD_Aug_{aug_method}.pth'
    )
