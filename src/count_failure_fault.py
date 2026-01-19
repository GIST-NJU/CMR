import argparse
import sys
from mr_utils import mrs
import pickle
import pandas as pd
import numpy as np
from itertools import permutations
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name, e.g. MNIST')
    parser.add_argument('--augment', type=str, default=None, choices=[None, 'no', 'online', 'offline'],
                        help="Use which augmented model for prediction (default: None, use original model), `None`, i.e. without --augment parameter, means using the checkpoint released by model author, `no` means no augmentation but trained by us")
    parser.add_argument('--cmr_num', type=int,
                        help="Number of CMR used in augmented model evaluation")
    parser.add_argument('--source_num', type=float,
                        help="Number of source inputs used in augmented model evaluation")
    parser.add_argument('--threshold', type=float, default=1, help='validity threshold for UTKFace')
    parser.add_argument('--bucket_size', type=float, default=1, help='bucket size for UTKFace fault definition')
    return parser.parse_args()

def validate_args(args):
    if args.dataset not in ['MNIST', 'Caltech256', 'VOC', 'COCO', 'UTKFace']:
        print(f"[failure] Dataset '{args.dataset}' is not sopported")
        print("Supported datasets: MNIST, caltech256, VOC, COCO, UTKFace")
        sys.exit(1)
    if args.augment is not None and args.cmr_num is None:
        print("[ERROR] Please specify --cmr_num when using augmented model for prediction")
        sys.exit(1)
    if args.augment and args.source_num is None:
        print("[ERROR] Please specify --source_num when using augmented model for prediction")
        sys.exit(1)
    if args.source_num and args.source_num >= 1:
        args.source_num = int(args.source_num)

def failure_revealing(dataset, mr, pred_source, pred_followup, validity_followup):
    if dataset in  ['MNIST', 'Caltech256', 'VOC', 'COCO']:
        failure = []
        for i in range(len(pred_source)):
            if validity_followup[mr][i] and (pred_followup[mr][i] != pred_source[i]):
                failure.append(i)
    elif dataset == 'UTKFace':
        failure = []
        for i in range(len(pred_source)):
            if validity_followup[mr][i] and abs(pred_followup[mr][i] - pred_source[i]) > utk_threshold:
                failure.append(i)
    else:
        print('Dataset not supported')
        sys.exit(1)
    return failure

strength_max = 5
models = {'MNIST': ['AlexNet'],
    'Caltech256': ['DenseNet121'],
    'VOC': ['MSRN'],
    'COCO': ['MLD'],
    'UTKFace': ['Faceptor']}
model_names = ['MNIST_AlexNet_9938', 'Caltech256_DenseNet121_6838', 'VOC_MSRN', 'COCO_MLD', 'UTKFace_Faceptor']
augmented_model_names = [
	'MNIST_AlexNet_Aug_online_9938', 'Caltech256_DenseNet121_Aug_online_7187', 'VOC_MSRN_Aug_online', 'COCO_MLD_Aug_online', 'UTKFace_Faceptor_Aug_online'
]


def get_model_name(dataset, model, augment=None):
	if not augment:
		for i in range(len(model_names)):
			if f"{dataset}_{model}" in model_names[i]:
				return model_names[i]
	else:
		for i in range(len(augmented_model_names)):
			if f"{dataset}_{model}_Aug_{augment}" in augmented_model_names[i]:
				return augmented_model_names[i]


def get_validity(dataset):
    filename_validity = 'results/validity/' + dataset + '_validity.npy'
    filename_threshold = 'results/validity/' + dataset + '_threshold.txt'
    validity = np.load(filename_validity, allow_pickle=True).item()
    with open(filename_threshold) as f:
        lines = f.readlines()
        threshold = float(lines[1].split(':')[1].strip())
    # print(threshold)
    for mr in validity:
        for i in range(len(validity[mr])):
            if validity[mr][i] <= threshold:
                validity[mr][i] = True
            else:
                validity[mr][i] = False
    return validity

def main():
    args = parse_args()
    validate_args(args)
    dataset = args.dataset
    global utk_threshold
    utk_threshold = args.threshold
    utk_bucket_size = args.bucket_size
    if args.source_num:
        with open(f'results/samples/{dataset}_{args.source_num}.pkl', 'rb') as f:
            selected_indices = pickle.load(f)
    else:
        selected_indices = None
    for index, model in enumerate(models[dataset]):
        model_name = get_model_name(dataset, model, args.augment)
        print(model_name)

        pred_followup_filename = f'{model_name}_followup' + (f'_{args.cmr_num}' if args.augment and args.cmr_num else '') + (f'_{args.source_num}' if args.augment and args.source_num else '')
        pred_source_filename = f'{model_name}_source' + (f'_{args.source_num}' if args.augment and args.source_num else '')

        print("Loading predictions...")
        pred_followup = np.load(os.path.join('results', 'predictions', dataset, pred_followup_filename+'.npy'),allow_pickle=True).item()
        if dataset in ['MNIST', 'Caltech256']:
            pred_source = np.load(os.path.join('results', 'predictions', dataset, pred_source_filename+'.npy'))
        elif dataset in ['VOC', 'COCO']:
            pred_source = pd.read_csv(os.path.join('results', 'predictions', dataset, pred_source_filename+'.csv'), low_memory=False)
            pred_source = pred_source.drop(columns=['img'])
            pred_source = pred_source.to_numpy()
            pred_source = {i: tuple(sorted(set(row[~pd.isna(row)]))) for i, row in enumerate(pred_source)}
            for key in pred_followup:
                t = pred_followup[key]
                t = t.drop(columns=['img'])
                t = t.to_numpy()
                t = {i: tuple(sorted(set(row[~pd.isna(row)]))) for i, row in enumerate(t)}
                pred_followup[key] = t
        elif dataset == 'UTKFace':
            pred_source = pd.read_csv(os.path.join('results', 'predictions', dataset, pred_source_filename+'.csv'), low_memory=False)
            pred_source = pred_source["pred_label"].to_numpy()
        else:
            print('Dataset not supported')
            sys.exit(1)

        failure = {}
        validity_followup = get_validity(dataset)

        if not args.augment and selected_indices is not None:
            print("Selecting predictions for specified inputs...")
            pred_source = [pred_source[i] for i in selected_indices]
            for mr in pred_followup:
                pred_followup[mr] = [pred_followup[mr][i] for i in selected_indices]

        if selected_indices is not None:
            for mr in tqdm(validity_followup, desc="Selecting validity for specified inputs"):
                validity_followup[mr] = [validity_followup[mr][i] for i in selected_indices]

        for cmr in tqdm(pred_followup, desc="Counting failures"):
            failure[cmr] = failure_revealing(dataset, cmr, pred_source, pred_followup, validity_followup)
        with open(f'results/errors/failure_{model_name}'+(f'_{args.source_num}' if args.source_num else '')+'.pkl', 'wb') as f:
            pickle.dump(failure, f)

        if dataset in ['MNIST', 'Caltech256', 'VOC', 'COCO']:
            faults_cmr = {}
            for cmr in tqdm(pred_followup, desc="Counting faults"):
                faults = {}
                for f in failure[cmr]:
                    source_label = pred_source[f]
                    followup_label = pred_followup[cmr][f]
                    faults[f] = (source_label, followup_label)
                faults_cmr[cmr] = faults
        elif dataset == 'UTKFace':
            faults_cmr = {}
            for cmr in tqdm(pred_followup, desc="Counting faults"):
                faults = {}
                for f in failure[cmr]:
                    source_label = pred_source[f]
                    followup_label = pred_followup[cmr][f]
                    faults[f] = (round(source_label / utk_bucket_size) * utk_bucket_size, round(followup_label / utk_bucket_size) * utk_bucket_size)
                faults_cmr[cmr] = faults
        else:
            print('Dataset not supported')
            sys.exit(1)
        with open(f'results/errors/fault_{model_name}'+(f'_{args.source_num}' if args.source_num else '')+'.pkl', 'wb') as f:
            pickle.dump(faults_cmr, f)
        print(model_name, 'Done')

if __name__ == "__main__":
    main()
