import os
import numpy as np
import cv2 as cv
from PIL import Image, ImageEnhance
from itertools import permutations
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

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


def generate(path, follow_path, cmr):
    source_path = path+'test2014'
    followup_path =  follow_path + 'followup'
    if not os.path.exists(followup_path):
        os.mkdir(followup_path)
    cmr_folder = ''.join([str(mr) for mr in cmr])
    if not os.path.exists(os.path.join(followup_path, cmr_folder)):
        os.mkdir(os.path.join(followup_path, cmr_folder))
    num = 0
    for imgname in os.listdir(source_path):
        num += 1
        img = Image.open(os.path.join(source_path, imgname))
        for index in cmr:
            img = mrs[index](img, paras[index])
        img.save(os.path.join(followup_path, cmr_folder, imgname.split('.')[0]+'.png'))
        print(num)
    if num % 100 == 0:
        print(num)

path = 'data/COCO/'
follow_path = '/fs14/home/hyw_husy/COCO/'
cmr = (1,)
start = time.time()
generate(path, follow_path, cmr)
end = time.time()
print((end - start) / 60)

img_dim = 128
batch_size = 2048

transform = transforms.Compose([
    transforms.Resize((img_dim, img_dim)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()])

def custom_collate(batch):
    inputs = [item[0] for item in batch]
    return default_collate(inputs)

test_set = datasets.CocoDetection(root=follow_path +'followup/'+''.join([str(mr) for mr in cmr]),
                            annFile=path + 'annotations/image_info_test2014_followup.json',
                            transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)