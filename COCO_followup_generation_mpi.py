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

def scale(x, scalar):
    height = int(x.size[0] * scalar)
    width = int(x.size[1] * scalar)
    dim = (width, height)
    return x.resize(dim)

def rotate(x, degree):
    # Rotate the image by degrees counter clockwise
    return x.rotate(degree)

def enh_bri(x, brightness):
    bri = ImageEnhance.Brightness(x)
    return bri.enhance(brightness)

def enh_con(x, contrast):
    con = ImageEnhance.Contrast(x)
    return con.enhance(contrast)

def enh_sha(x, sharpness):
    sha = ImageEnhance.Sharpness(x)
    return sha.enhance(sharpness)

def gaussian(x, kernel_size):
    x = np.array(x)
    x = cv.GaussianBlur(x, kernel_size, sigmaX=0)
    return Image.fromarray(x)

def shear(x, shear_factor):
    # 定义错切变换矩阵
    shear_matrix = [1, shear_factor, 0, 0, 1, 0]

    # 创建Affine对象并应用错切变换
    sheared_img = x.transform(
        x.size, Image.Transform.AFFINE, shear_matrix
    )
    return sheared_img

def translate(x, shift):
    shift_x, shift_y = shift[0], shift[1]
    # 进行平移操作
    translated_img = x.transform(
        x.size, Image.Transform.AFFINE, (1, 0, shift_x, 0, 1, shift_y)
    )
    return translated_img

mrs = [rotate, enh_bri, enh_sha, enh_con, gaussian, shear, translate]
paras = [3, 0.8, 0.8, 0.8, (3, 3), 0.1, (1,1)]



def generate(path, followuppath, cmr):
    source_path = path + 'test2014'
    print(len(os.listdir(source_path)))
    followuppath = followuppath + 'followup'
    if not os.path.exists(followuppath):
        os.mkdir(followuppath)
    cmr_folder = ''.join([str(mr) for mr in cmr])
    cmr_path = os.path.join(followuppath, cmr_folder)
    if not os.path.exists(cmr_path):
        os.mkdir(cmr_path)
    file_names = os.listdir(cmr_path)
    num = 0
    for imgname in os.listdir(source_path):
        num += 1
        #print(num, cmr_path)
        if imgname.split('.')[0]+'.png' in file_names:
            continue
        print(num, cmr_path)
        img = Image.open(os.path.join(source_path, imgname))
        for index in cmr:
            img = mrs[index](img, paras[index])
        img.save(os.path.join(cmr_path, imgname.split('.')[0]+'.png'))        

path = 'data/COCO/'
followuppath = '/fs14/home/hyw_husy/COCO/'

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if rank == 0:
    all_tasks = [cmr for i in range(len(mrs)) for cmr in permutations(range(len(mrs)), i+1)]

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
        generate(path, followuppath, cmr)
        end = time.time()
        print(f'Rank {rank}:', cmr, (end - start) / 60)
print('Done!')