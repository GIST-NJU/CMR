# mr_utils.py
from PIL import Image, ImageEnhance
import cv2 as cv
import numpy as np

__all__ = [
    "rotate",
    "enh_bri",
    "enh_con",
    "enh_sha",
    "gaussian",
    "shear",
    "translate",
    "mrs",
    "mrs_name",
    "paras",
]

def rotate(x, degree):
    return x.rotate(degree)

def enh_bri(x, brightness):
    return ImageEnhance.Brightness(x).enhance(brightness)

def enh_con(x, contrast):
    return ImageEnhance.Contrast(x).enhance(contrast)

def enh_sha(x, sharpness):
    return ImageEnhance.Sharpness(x).enhance(sharpness)

def gaussian(x, kernel_size):
    x_np = np.array(x)
    blurred = cv.GaussianBlur(x_np, kernel_size, sigmaX=0)
    return Image.fromarray(blurred)

def shear(x, shear_factor):
    return x.transform(x.size, Image.Transform.AFFINE,
                       [1, shear_factor, 0, 0, 1, 0])

def translate(x, shift):
    dx, dy = shift
    return x.transform(x.size, Image.Transform.AFFINE,
                       [1, 0, dx, 0, 1, dy])

mrs = [rotate, enh_bri, enh_sha, enh_con, gaussian, shear, translate]
mrs_name = [f.__name__ for f in mrs]
paras = [3, 0.8, 0.8, 0.8, (3, 3), 0.1, (1, 1)]