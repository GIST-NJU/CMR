import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from mr_utils import mrs

import pickle
import os
import random
from itertools import permutations
import argparse


strength_max = 5
model_names = ['MNIST_AlexNet_9938', 'Caltech256_DenseNet121_6838', 'VOC_MSRN', 'COCO_MLD', 'UTKFace_Faceptor']
test_cases_num = {'MNIST': 10000, 'Caltech256': 3061, 'VOC': 4952, 'COCO': 40775, 'UTKFace': 3287}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmr_num', type=int, required=True,
                        help="Number of CMR used in augmented model evaluation")
    parser.add_argument('--source_num', type=float, required=True,
                        help="Number of source inputs used in augmented model evaluation")
    args = parser.parse_args()
    if args.source_num < 0:
        print("[ERROR] source_num must be >= 0")
        sys.exit(1)
    if args.source_num >= 1:
        args.source_num = int(args.source_num)
    return args


def select_cmrs_for_augmentation(model_name, method='random', num=10, strength_max=5):
    mr_num = len(mrs)
    if method == 'random':
        all_mrs = [cmr for i in range(2, strength_max+1) for cmr in permutations(range(mr_num), i)]
        selected_cmrs = random.sample(all_mrs, num)
    elif method == 'failure_rate':
        with open(f'results/errors/failure_{model_name}.pkl', 'rb') as f:
            failures = pickle.load(f)
        all_mrs = []
        for i in range(2, strength_max+1):
            for cmr in permutations(range(mr_num), i):
                fr = len(failures[cmr])
                all_mrs.append((cmr, fr))
        all_mrs = sorted(all_mrs, key=lambda x: x[1], reverse=True)
        selected_cmrs = [all_mrs[i][0] for i in range(num)]
    elif method == 'fault_type':
        with open(f'results/errors/fault_{model_name}.pkl', 'rb') as f:
            faults = pickle.load(f)
        all_mrs = []
        for i in range(2, strength_max+1):
            for cmr in permutations(range(mr_num), i):
                ft = len(set(faults[cmr]))
                all_mrs.append((cmr, ft))
        all_mrs = sorted(all_mrs, key=lambda x: x[1], reverse=True)
        selected_cmrs = [all_mrs[i][0] for i in range(num)]
    return selected_cmrs


if __name__ == "__main__":
    if not os.path.exists('results/samples'):
        os.makedirs('results/samples')
    args = parse_args()
    random.seed(42)
    for model_name in model_names:
        print(f"Model: {model_name}")

        cmr_save_path = f'results/samples/{model_name}_cmr{args.cmr_num}.pkl'

        if os.path.exists(cmr_save_path):
            print(f"{cmr_save_path} already exists, skipping...")
            continue

        # select cmrs
        selected_cmrs = {}
        # single cmr
        selected_cmrs["single_cmr"] = [(i,) for i in range(len(mrs))]
        # top-5 cmrs by failure rate
        selected_cmrs["top_failure_rate"] = select_cmrs_for_augmentation(model_name, method='failure_rate', num=args.cmr_num, strength_max=strength_max)
        # random 5 for 5 times
        for i in range(5):
            selected_cmrs[f"random_{i}"] = select_cmrs_for_augmentation(model_name, method='random', num=args.cmr_num, strength_max=strength_max)

        print(selected_cmrs)
        with open(cmr_save_path, 'wb') as f:
            pickle.dump(selected_cmrs, f)

    random.seed(24)
    for model_name in model_names:
        print(f"Model: {model_name}")
        source_save_path = f'results/samples/{model_name.split("_")[0]}_{args.source_num}.pkl'

        if os.path.exists(source_save_path):
            print(f"{source_save_path} already exists, skipping...")
            continue

        # select source images
        if args.source_num == 0:
            select_source_images = None
        if args.source_num < 1:
            total_cases = test_cases_num[model_name.split('_')[0]]
            select_num_cases = int(total_cases * args.source_num)
            select_source_images = sorted(random.sample(range(test_cases_num[model_name.split('_')[0]]), select_num_cases))
        else:
            select_num_cases = min(args.source_num, test_cases_num[model_name.split('_')[0]])
            select_source_images = sorted(random.sample(range(test_cases_num[model_name.split('_')[0]]), min(test_cases_num[model_name.split('_')[0]], select_num_cases)))

        print(select_source_images, f"len={len(select_source_images)}")
        with open(source_save_path, 'wb') as f:
            pickle.dump(select_source_images, f)
