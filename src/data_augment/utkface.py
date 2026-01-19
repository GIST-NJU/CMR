import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'UTKFace', 'Faceptor')))
from models.UTKFace.Faceptor.core.utils import setup_seed, worker_init_fn
from models.UTKFace.Faceptor.core.model.backbone import FaRLVisualFeatures
from models.UTKFace.Faceptor.core.model.heads import DecoderNewHolder
from models.UTKFace.Faceptor.core.model.loss import LossesHolder
from models.UTKFace.Faceptor.core.model.model_entry import AIOEntry

from models.UTKFace.Faceptor.core.data.transform import transform_entry
from models.UTKFace.Faceptor.core.optimizer import optimizer_entry
from models.UTKFace.Faceptor.core.lr_scheduler import lr_scheduler_entry
from models.UTKFace.Faceptor.core.data.sampler import sampler_entry
from models.UTKFace.Faceptor.core.utils import LimitedAvgMeter
from models.UTKFace.Faceptor.core.utils import AverageMeter as LengthAvgMeter
from models.UTKFace.Faceptor.core.data.dataloader import DataLoaderX

from dataloaders.utkface import UTKFaceDataset_v3, NumpyToTensorRawTransform
from .utils import DataAugmentationDataset

from easydict import EasyDict as edict
from functools import partial
import torch
from torch import distributed
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.utils import AverageMeter
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import copy
import time
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


cfg = edict({
    "seed": 2048,
    "cuda_deterministic": False,
    "fp16": True,
    "num_workers": 20,
    "max_iter": 50000,
    "last_iter": -1,
    "gradient_acc": 1,
    "optimizer": {
        "type": "AdamW",
        "kwargs": {
            "weight_decay": 0.05
        }
    },
    "backbone_multiplier": 1.0,
    "heads_multiplier": 10.0,
    "interpreters_multiplier": 10.0,
    "decoder_multiplier": 10.0,
    "losses_multiplier": 10.0,
    "lr_scheduler": {
        "type": "Cosine",
        "kwargs": {
            "eta_min": 0.0,
            "base_lr": 1.e-6,
            "warmup_lr": 5.e-5,
            "warmup_steps": 2000
        }
    },
    "print_interval": 10,
    "save_interval": 5000,
    "evaluate_interval": 10000,
})


utkface_test_task = edict({
    "name": "age_utkface",
    "loss_weight": 1.0,
    "dataset": {
        "type": "UTKFaceDataset_v3",
        "kwargs": {
            "data_path": "data/source/UTKFace/UTK_train_selected",
            "augmentation": {
                "type": "attribute_train_transform",
                "kwargs": {
                    "input_size": 112,
                    "mean": [0.48145466, 0.4578275, 0.40821073],
                    "std": [0.26862954, 0.26130258, 0.27577711]
                }
            },
            "train": True,
        }
    },
    "sampler": {
        "type": "DistributedGivenIterationSampler",
        "batch_size": 128,
        "shuffle_strategy": 1
    },
    "loss": {
        "type": "AgeLoss_DLDLV2",
        "kwargs": {}
    },
    "evaluator": {
        "type": "AgeEvaluator_V2",
        "use": True,
        "kwargs": {
            "mark": "eval_utkface_test_source",
            "test_batch_size": 32,
            "test_dataset_cfg": {
                "type": "UTKFaceDataset_v3",
                "kwargs": {
                    "data_path": "data/source/UTKFace/UTK_test_selected",
                    "augmentation": {
                        "type": "attribute_test_transform",
                        "kwargs": {
                            "input_size": 112,
                            "mean": [0.48145466, 0.4578275, 0.40821073],
                            "std": [0.26862954, 0.26130258, 0.27577711]
                        }
                    },
                    "train": False,
                }
            }
        }
    }
})


class AgeEvaluator_V3(object):

    def __init__(self, dataset, test_batch_size, mark):
        self.mark = mark

        self.mae_lowest = 100.0
        self.cs_highest = 0.0
        self.eps_error_lowest = 100.0

        self.dataset = dataset
        self.data_loader = DataLoader(
            self.dataset, batch_size=test_batch_size, 
            shuffle=False, pin_memory=True, drop_last=False)
        
        self.mae_lmeter = LimitedAvgMeter(max_num=10, best_mode="min")
        self.cs_lmeter=LimitedAvgMeter(max_num=10, best_mode="max")
        self.eps_error_lmeter=LimitedAvgMeter(max_num=10, best_mode="min")

    def ver_test(self, model):
        print(self.mark)
        mae_meter = AverageMeter()
        cs_meter = AverageMeter()
        eps_error_meter = AverageMeter()

        eval_results = []

        for idx, input_var in tqdm(enumerate(self.data_loader), desc=f'Evaluating {self.mark}', total=len(self.data_loader)):

            input_var["image"]=input_var["image"].cuda()
            out_var = model(input_var)
            age_output = out_var["head_output"]


            age_output = F.sigmoid(age_output)
            age_output = F.normalize(age_output, p=1, dim=1)


            label = input_var["label"]
            std_label = 3.0
            avg_label = label["avg_label"].numpy()


            rank = torch.Tensor([i for i in range(101)]).cuda()
            age_output = torch.sum(age_output*rank, dim=1)

            age_output = age_output.cpu().detach().numpy()

            for filename, pred in zip(input_var["filename"], age_output):
                eval_results.append({'filename': os.path.basename(filename), 'pred_label': pred})

            mae = np.array(abs(age_output - avg_label), dtype=np.float32).mean()
            cs = np.array(abs(age_output - avg_label) <= 5, dtype=np.float32).mean()*100
            eps_error = 1 - np.mean((1 / np.exp(np.square(np.subtract(age_output, avg_label)) / (2 * np.square(std_label)))))

            mae_meter.update(mae, age_output.shape[0])
            cs_meter.update(cs, age_output.shape[0])
            eps_error_meter.update(eps_error, age_output.shape[0])

        if mae_meter.avg < self.mae_lowest:
            self.mae_lowest = mae_meter.avg
        if cs_meter.avg > self.cs_highest:
            self.cs_highest = cs_meter.avg
        if eps_error_meter.avg < self.eps_error_lowest:
            self.eps_error_lowest = eps_error_meter.avg

        self.mae_lmeter.append(mae_meter.avg)
        self.cs_lmeter.append(cs_meter.avg)
        self.eps_error_lmeter.append(eps_error_meter.avg)

        print(f'[{self.mark}] MAE: {mae_meter.avg:.6f}')
        print(f'[{self.mark}] MAE-Lowest: {self.mae_lowest:.6f}')
        print(f'[{self.mark}] MAE-Mean@10: {self.mae_lmeter.avg:.6f}')
        print(f'[{self.mark}] CS: {cs_meter.avg:.6f}')
        print(f'[{self.mark}] CS-Highest: {self.cs_highest:.6f}')
        print(f'[{self.mark}] CS-Mean@10: {self.cs_lmeter.avg:.6f}')
        print(f'[{self.mark}] Eps-Error: {eps_error_meter.avg:.6f}')
        print(f'[{self.mark}] Eps-Error-Lowest: {self.eps_error_lowest:.6f}')
        print(f'[{self.mark}] Eps-Error-Mean@10: {self.eps_error_lmeter.avg:.6f}')

        self.eval_results_df = pd.DataFrame(eval_results)
        self.eval_results_df.sort_values(by=['filename'], inplace=True, ignore_index=True)
        return mae_meter.avg


    def __call__(self, num_update, model):
        if num_update > 0:
            self.eval_results_df = None
            model.eval()
            mae = self.ver_test(model)
            torch.cuda.empty_cache()
            return self.eval_results_df, mae


def load_dataset(aug_method):
    trainset = UTKFaceDataset_v3(
        data_path=utkface_test_task['dataset']['kwargs']['data_path'],
        augmentation=utkface_test_task['dataset']['kwargs']['augmentation'],
        train=True,
        transform=transforms.Compose([
            transform_entry(utkface_test_task['dataset']['kwargs']["augmentation"]),
            NumpyToTensorRawTransform()
        ])
    )
    valset = UTKFaceDataset_v3(
        data_path=utkface_test_task['evaluator']['kwargs']['test_dataset_cfg']['kwargs']['data_path'],
        augmentation=utkface_test_task['evaluator']['kwargs']['test_dataset_cfg']['kwargs']['augmentation'],
        train=False,
        transform=transforms.Compose([
            transform_entry(utkface_test_task['evaluator']['kwargs']["test_dataset_cfg"]["kwargs"]["augmentation"]),
            NumpyToTensorRawTransform()
        ])
    )
    if aug_method is None or aug_method == 'no':
        return trainset, valset

    train_transform = trainset.transform
    trainset.transform = None
    augmented_trainset = DataAugmentationDataset(
        dataname='UTKFace',
        dataset=trainset,
        get_img_from_item_fn=lambda item: item['image'],
        prepare_item_fn=lambda img, item: ({'image': img, 'label': item['label'], 'filename': item['filename']}),
        aug_method=aug_method,
        transform=train_transform,
        pre_generate=False,
        save_followup=False,
    )
    return augmented_trainset, valset


def load_model():
    # create model
    backbone = FaRLVisualFeatures(
        model_type="base",
        model_path="models/UTKFace/Faceptor/FaRL-Base-Patch16-LAIONFace20M-ep64.pth",
        drop_path_rate=0.2,
        forced_input_resolution=512
    )
    heads_holder = DecoderNewHolder(
        task_names=["age_utkface"],
        query_nums=[101],
        interpreter_types=["value"],
        out_types=["None"],
        decoder_type="TransformerDecoderLevel",
        levels=[11]
    )
    task_cfgs = {0: utkface_test_task}
    model_entry_cfg = edict({
        "type": "aio_entry",
        "kwargs": {
            "size_group": {
                "group_0": {
                    "task_types": ["age"],
                    "input_size": 112
                },
                # "group_0": {
                #     "task_types": ["recog", "age", "biattr", "affect"],
                #     "input_size": 112
                # },
                # "group_1": {
                #     "task_types": ["parsing", "align"],
                #     "input_size": 512
                # }
            }
        }
    })
    losses_holder = LossesHolder(task_cfgs)
    model = AIOEntry(model_entry_cfg, task_cfgs, backbone, heads_holder, losses_holder)
    model.to(device)
    print("model: ", model)
    model = torch.nn.parallel.DistributedDataParallel(module=model, broadcast_buffers=False, 
                                                        device_ids=[0], bucket_cap_mb=16,
                                                        find_unused_parameters=True)
    model.register_comm_hook(None, fp16_compress_hook)
    return model


def create_optimizer(model):
    # create optimizer
    defaults = {
        "lr": cfg.lr_scheduler.kwargs.base_lr,
        "weight_decay": cfg.optimizer.kwargs.weight_decay
    }
    memo = set()
    param_groups = []

    for module_name, module in model.named_modules():
        for _, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            # Set learning rate.
            hyperparams = copy.copy(defaults)
            if "backbone" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * cfg.get('backbone_multiplier', 1.0)
            if "heads" in module_name:
                if "interpreters" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.get('interpreters_multiplier', 1.0)
                elif "decoder" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.get('decoder_multiplier', 1.0)
                else:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.get('heads_multiplier', 1.0)
            if "losses_module" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * cfg.get('losses_multiplier', 1.0)

            param_groups.append({"params": [value], **hyperparams})


    cfg.optimizer.kwargs.params = param_groups
    cfg.optimizer.kwargs.lr = cfg.lr_scheduler.kwargs.base_lr
    optimizer = optimizer_entry(cfg.optimizer)
    return optimizer


def create_lr_scheduler(optimizer):
    cfg.lr_scheduler.kwargs.optimizer = optimizer
    cfg.lr_scheduler.kwargs.last_iter = cfg.last_iter
    cfg.lr_scheduler.kwargs.max_iter = cfg.max_iter
    lr_scheduler = lr_scheduler_entry(cfg.lr_scheduler)
    return lr_scheduler


def create_dataloader(trainset):
    train_sampler = sampler_entry(utkface_test_task.sampler)(
        dataset=trainset,
        task_name=utkface_test_task.name,
        total_iter=cfg.max_iter,
        batch_size=utkface_test_task.sampler.batch_size, 
        world_size=world_size, 
        rank=rank,
        last_iter=cfg.last_iter,
        shuffle_strategy=utkface_test_task.sampler.shuffle_strategy, 
        random_seed=cfg.seed,
        ret_save_path=utkface_test_task.sampler.get('ret_save_path', None)
    )
    init_fn = partial(worker_init_fn, num_workers=cfg.num_workers, rank=rank, seed=cfg.seed)
    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=trainset,
        batch_size=utkface_test_task.sampler.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )
    return train_loader


def logging(tmp, output, loss):
    tmp.loss_total.update(loss.item())
    i = 0
    task_name = utkface_test_task.name
    tmp.loss_list[i].update(output[task_name]["tloss"].item())
    tmp.top1_list[i].update(output[task_name]["top1"].item())

    if (tmp.current_step + 1) % cfg.get('print_interval', 10) == 0:
        log_msg = '\t'.join([
            'Iter: [{0}/{1}] ',
            'Time: {batch_time.avg:.3f} (ETA:{eta:.2f}h) ({data_time.avg:.3f}) ',
            # 'Total Loss: {loss.avg:.4f} ',
            'LR: {current_lr} ',
            '{meters} ',
            'max mem: {memory:.0f}'
        ])

        MB = 1024.0 * 1024.0

        loss_str = []
        loss_str.append(
            "{}_loss(top1): {:4f}({:4f}) ".format(utkface_test_task.name, tmp.loss_list[i].avg, tmp.top1_list[i].avg)
        )
        loss_str = '\t'.join(loss_str)
        log_msg = log_msg.format(tmp.current_step, cfg.max_iter, \
                        batch_time=tmp.batch_time, \
                        eta=(cfg.max_iter-tmp.current_step)*tmp.batch_time.avg/3600, \
                        data_time=tmp.data_time, \
                        # loss=tmp.loss_total, \
                        current_lr=tmp.current_lr, \
                        meters=loss_str, \
                        memory=torch.cuda.max_memory_allocated() / MB)
        
        log_msg+="\n"
        print(log_msg)


def save(tmp, model, optimizer, mae):
    checkpoint = {
        'step': tmp.current_step+1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_path = os.path.join('models', 'UTKFace', 'Faceptor', 'augmented')
    os.makedirs(ckpt_path, exist_ok=True)

    if (tmp.current_step + 1) % cfg.get('ckpt_interval', 1000) == 0:
        torch.save(checkpoint, os.path.join(ckpt_path, "checkpoint_iter_newest.pth.tar"))

    if cfg.get('save_interval', -1) > 0 and ((tmp.current_step + 1) % cfg.save_interval == 0 or tmp.current_step + 1 == cfg.max_iter):
        torch.save(checkpoint, os.path.join(ckpt_path, f"checkpoint_iter_{tmp.current_step + 1}.pth.tar"))
    
    if mae is not None:
        if tmp.mae_best < 0 or mae < tmp.mae_best:
            tmp.mae_best = mae
            torch.save(checkpoint, os.path.join(ckpt_path, "checkpoint_iter_best_mae.pth.tar"))


def evaluate(tmp, model, evaluator):
    if cfg.get('evaluate_interval', -1) > 0 and ((tmp.current_step + 1) % cfg.evaluate_interval == 0 or tmp.current_step + 1 == cfg.max_iter):
        model.module.set_mode_to_evaluate()

        model.module.set_evaluation_task(utkface_test_task.name)
        _, mae = evaluator(tmp.current_step, model)

        model.module.set_mode_to_train()
        return mae
    return None


def run(aug_method):
    os.makedirs('models', exist_ok=True)

    trainset, valset = load_dataset(aug_method)
    
    setup_seed(cfg.seed, cfg.cuda_deterministic)
    # prepare
    model = load_model()
    optimizer = create_optimizer(model)
    lr_scheduler = create_lr_scheduler(optimizer)
    dataloader = create_dataloader(trainset)
    evaluator = AgeEvaluator_V3(
        dataset = valset,
        test_batch_size = utkface_test_task.evaluator.kwargs.test_batch_size,
        mark = "UTKFace While Training"
    )
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    tmp = edict()
    tmp.data_time = LengthAvgMeter(10)
    tmp.batch_time = LengthAvgMeter(10)
    tmp.loss_total = LengthAvgMeter(10)

    tmp.loss_list = [LengthAvgMeter(10)]
    tmp.top1_list = [LengthAvgMeter(10)]
    tmp.mae_best = -1

    # train
    model.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    model._set_static_graph()
    
    end = time.time()

    for i, data in enumerate(zip(*[dataloader])):
        tmp.current_step = cfg.last_iter + i + 1
        tmp.data_time.update(time.time() - end)
        
        loss, output = model(data, current_step=tmp.current_step)


        if cfg.fp16:
            amp.scale(loss).backward()
            if tmp.current_step % cfg.gradient_acc == 0:
                amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                amp.step(optimizer)
                amp.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if tmp.current_step % cfg.gradient_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()
        lr_scheduler.step(tmp.current_step)
        tmp.current_lr = lr_scheduler.get_lr()[0]


        tmp.batch_time.update(time.time() - end)
        end = time.time()

        with torch.no_grad():
            logging(tmp, output, loss)
            mae = evaluate(tmp, model, evaluator)

        save(tmp, model, optimizer, mae)

    shutil.copy(
        os.path.join('models', 'UTKFace', 'Faceptor', 'augmented', 'checkpoint_iter_newest.pth.tar'),
        os.path.join('models', f'UTKFace_Faceptor_Aug_{aug_method}.pth.tar')
    )
