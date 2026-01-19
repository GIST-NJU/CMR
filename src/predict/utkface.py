import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'UTKFace', 'Faceptor')))

from models.UTKFace.Faceptor.core.utils import setup_seed, load_state_model
from models.UTKFace.Faceptor.core.model.backbone import FaRLVisualFeatures
from models.UTKFace.Faceptor.core.model.heads import DecoderNewHolder
from models.UTKFace.Faceptor.core.model.loss import LossesHolder
from models.UTKFace.Faceptor.core.model.model_entry import AIOEntry

from models.UTKFace.Faceptor.core.data.transform import transform_entry
from models.UTKFace.Faceptor.core.utils import LimitedAvgMeter

from dataloaders.utkface import UTKFaceDataset_v3, NumpyToTensorRawTransform

from easydict import EasyDict as edict
import torch
from torch import distributed
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.utils import AverageMeter
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import gc
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
import random


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
        init_method=f"tcp://127.0.0.1:{random.randint(10000, 20000)}",
        rank=rank,
        world_size=world_size,
    )

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
        "batch_size": 32,
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


    def __call__(self, model):
        self.eval_results_df = None
        model.eval()
        self.ver_test(model)
        torch.cuda.empty_cache()
        return self.eval_results_df


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
    source_testset = UTKFaceDataset_v3(
        data_path=utkface_test_task['evaluator']['kwargs']['test_dataset_cfg']['kwargs']['data_path'],
        augmentation=utkface_test_task['evaluator']['kwargs']['test_dataset_cfg']['kwargs']['augmentation'],
        train=False,
        transform=transforms.Compose([
            transform_entry(utkface_test_task['evaluator']['kwargs']["test_dataset_cfg"]["kwargs"]["augmentation"]),
            NumpyToTensorRawTransform()
        ])
    )
    if source_num:
        with open(f'results/samples/UTKFace_{source_num}.pkl', 'rb') as f:
            selected_indices = pickle.load(f)
        source_testset = Subset(source_testset, selected_indices)
    evalautor = AgeEvaluator_V3(
        dataset=source_testset,
        test_batch_size=utkface_test_task['evaluator']['kwargs']['test_batch_size'],
        mark=f"{model_name} Source",
    )
    res_df = evalautor(model)
    res_df.to_csv(save_path, index=False)


def test_followup(augment, cmr_num=None, source_num=None):
    save_path = os.path.join(folder_path, model_name+'_followup'+(f'_{cmr_num}' if cmr_num else '')+(f'_{source_num}' if source_num else '')+'.npy')
    if os.path.exists(save_path):
        print("Followup predictions already exist.")
        return
    pred_followup = {}
    followup_dir = 'data/followup/UTKFace'
    if not augment:
        entries = os.listdir(followup_dir)
    else:
        with open(f'results/samples/UTKFace_Faceptor_cmr{cmr_num}.pkl', 'rb') as f:
            selected_cmrs = pickle.load(f)
        entries = [''.join(map(str, cmr)) for cmrs in selected_cmrs.values() for cmr in cmrs]
    folders = [entry for entry in entries if os.path.isdir(os.path.join(followup_dir, entry))]
    folders = sorted(folders)
    if source_num:
        with open(f'results/samples/UTKFace_{source_num}.pkl', 'rb') as f:
            selected_indices = pickle.load(f)
    else:
        selected_indices = None
    for folder in tqdm(folders, desc="Predicting followup inputs"):
        cmr = tuple(int(char) for char in folder)
        followup_path = os.path.join(followup_dir, folder)
        follow_testset = UTKFaceDataset_v3(
            data_path=followup_path,
            augmentation=utkface_test_task['evaluator']['kwargs']['test_dataset_cfg']['kwargs']['augmentation'],
            train=False,
            transform=transforms.Compose([
                transform_entry(utkface_test_task['evaluator']['kwargs']["test_dataset_cfg"]["kwargs"]["augmentation"]),
                NumpyToTensorRawTransform()
            ]),
            selected_indices=selected_indices
        )
        evalautor = AgeEvaluator_V3(
            follow_testset,
            test_batch_size=utkface_test_task['evaluator']['kwargs']['test_batch_size'],
            mark=f"{model_name} Followup"
        )
        res_df = evalautor(model)
        res = res_df['pred_label'].to_numpy()
        pred_followup[cmr] = res
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
    np.save(save_path, pred_followup)


def load_model(augment):
    global model, model_name, folder_path

    if not augment:
        model_name = 'UTKFace_Faceptor'
    else:
        model_name = f'UTKFace_Faceptor_Aug_{augment}'
    folder_path = './results/predictions/UTKFace'
    os.makedirs(folder_path, exist_ok=True)

    setup_seed(2048, False)
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
                    "task_types": ["recog", "age", "biattr", "affect"],
                    "input_size": 112
                },
                "group_1": {
                    "task_types": ["parsing", "align"],
                    "input_size": 512
                }
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

    ckpt_path = f'models/{model_name}.pth.tar'
    try:
        checkpoint = torch.load(ckpt_path, 'cpu')
    except Exception as e:
        raise FileNotFoundError(f'=> no checkpoint found at {ckpt_path}')
    print(f"Recovering from {ckpt_path}, keys={list(checkpoint.keys())}")
    pretrained_state_dict = checkpoint['state_dict']
    load_state_model(model, pretrained_state_dict)
    model.module.set_mode_to_evaluate()
    model.module.set_evaluation_task("age_utkface")


def run(followup, augment, cmr_num=None, source_num=None):
    load_model(augment)
    if not followup:
        test_source(source_num)
    else:
        test_followup(augment, cmr_num, source_num)
