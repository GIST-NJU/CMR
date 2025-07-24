import os
import numpy as np
import cv2 as cv
from PIL import Image, ImageEnhance
from itertools import permutations
import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import boto3
from io import BytesIO
from mpi4py import MPI

def check_images(directory_path):
    corrupted_images = []
    for filename in os.listdir(directory_path):
        #print(filename)
        file_path = os.path.join(directory_path, filename)
        try:
            with Image.open(file_path) as img:
                img.verify()  # 验证图像
        except (IOError, SyntaxError) as e:
            corrupted_images.append(filename)
    return corrupted_images


path = '/fs14/home/hyw_husy/COCO/followup'

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if rank == 0:
    all_tasks = [cmr for i in range(7) for cmr in permutations(range(7), i+1)]

    chunk_size = len(all_tasks) // size + (1 if len(all_tasks) % size > 0 else 0)
    tasks = [all_tasks[i:i + chunk_size] for i in range(0, len(all_tasks), chunk_size)]
    while len(tasks) < size:
        tasks.append([])

else:
    tasks = None

tasks = comm.scatter(tasks, root=0)

for task in tasks:
    if task:
        cmr = task
        start = time.time()
        corrupted = check_images(os.path.join(path, "".join(str(j) for j in cmr)))
        end = time.time()
        print(f'Rank {rank}:', cmr, round((end - start) / 60, 2), corrupted)
print('Done!')