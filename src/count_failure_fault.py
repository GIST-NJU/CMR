import argparse
import sys
from mr_utils import mrs
import pickle
import pandas as pd
import numpy as np
from itertools import permutations
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name, e.g. MNIST')
    return parser.parse_args()

def validate_args(args):
    if args.dataset not in ['MNIST', 'caltech256', 'VOC', 'COCO']:
        print(f"[failure] Dataset '{args.dataset}' is not sopported")
        print("Supported datasets: MNIST, caltech256, VOC, COCO")
        sys.exit(1)

def failure_revealing(dataset, mr, pred_source, pred_followup, validity_followup):
    if dataset in  ['MNIST', 'Caltech256']:
        failure = []
        for i in range(len(pred_source)):
            if validity_followup[mr][i] and (pred_followup[mr][i] != pred_source[i]):
                failure.append(i)
    else:
        pred_f = pred_followup[mr]
        pred_f = pred_f.drop(columns=['img'])
        pred_f = pred_f.to_numpy()
        pred_f = {i: set(row[~pd.isna(row)]) for i, row in enumerate(pred_f)}
        failure = []
        for i in range(len(pred_source)):
            if validity_followup[mr][i] and (pred_f[i] != pred_source[i]):
                failure.append(i)
    return failure

strength_max = 5
models = {'MNIST': ['AlexNet'],
    'Caltech256': ['DenseNet121'],
    'VOC': ['MSRN'],
    'COCO': ['MLD']}
model_names = ['MNIST_AlexNet_9938', 'Caltech256_DenseNet121_6838', 'VOC_MSRN', 'COCO_MLD']

def get_model_name(dataset, model):
    for i in range(len(model_names)):
        if dataset+'_'+model in model_names[i]:
            return model_names[i]

def get_validity(dataset):
    filename_validity = 'results/SelfOracle/' + dataset + '_validity.npy'
    filename_threshold = 'results/SelfOracle/' + dataset + '_threshold.txt'
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
    dataset = args.dataset
    for index, model in enumerate(models[dataset]):
        model_name = get_model_name(dataset, model)
        pred_followup = np.load(os.path.join('results', 'predictions', dataset, model_name+'_followup.npy'),allow_pickle=True).item()
        if dataset in ['MNIST', 'Caltech256']:
            pred_source = np.load(os.path.join('results', 'predictions', dataset, model_name+'_source.npy'))
        else:
            pred_source = pd.read_csv(os.path.join('results', 'predictions', dataset, model_name+'_source.csv'), low_memory=False)
            pred_source = pred_source.drop(columns=['img'])
            pred_source = pred_source.to_numpy()
            pred_source = {i: tuple(sorted(set(row[~pd.isna(row)]))) for i, row in enumerate(pred_source)}
            for key in pred_followup:
                t = pred_followup[key]
                t = t.drop(columns=['img'])
                t = t.to_numpy()
                t = {i: tuple(sorted(set(row[~pd.isna(row)]))) for i, row in enumerate(t)}
                pred_followup[key] = t

        failure = {}
        validity_followup = get_validity(dataset)
        for mr in pred_followup:
            failure[mr] = failure_revealing(dataset, mr, pred_source, pred_followup, validity_followup)
        with open(f'results/errors/failure_{model_name}.pkl', 'wb') as f:
            pickle.dump(failure, f)

        faults_cmr = {}
        for i in range(strength_max):
            for cmr in permutations(range(len(mrs)), i + 1):
                faults = {}
                for f in failure[cmr]:
                    source_label = pred_source[f]
                    followup_label = pred_followup[cmr][f]
                    faults[f] = (source_label, followup_label)
                faults_cmr[cmr] = faults
        with open(f'results/errors/fault_{model_name}.pkl', 'wb') as f:
            pickle.dump(faults_cmr, f)
        print(dataset, model, 'Done')

if __name__ == "__main__":
    main()