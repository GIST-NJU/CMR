import os
import numpy as np
import cv2 as cv
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms,models
import torch.backends.cudnn as cudnn
from itertools import permutations
from sklearn.model_selection import train_test_split
import math

from torch.utils.data.dataloader import default_collate

torch.manual_seed(28)
cudnn.deterministic = True

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
    shear_matrix = [1, shear_factor, 0, 0, 1, 0]

    sheared_img = x.transform(
        x.size, Image.Transform.AFFINE, shear_matrix
    )
    return sheared_img

def translate(x, shift):
    shift_x, shift_y = shift[0], shift[1]

    translated_img = x.transform(
        x.size, Image.Transform.AFFINE, (1, 0, shift_x, 0, 1, shift_y)
    )
    return translated_img

mrs = [rotate, enh_bri, enh_sha, enh_con, gaussian, shear, translate]
mrs_name =[mr.__name__ for mr in mrs]
paras = [3, 0.8, 0.8, 0.8, (3, 3), 0.1, (1,1)]

class CustomDataset(Dataset):
    def __init__(self, dataset, cmr=None, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        for idx, (img, label) in enumerate(dataset):
            if cmr is not None:
                for index in cmr:
                    img = mrs[index](img, paras[index])
            self.data.append(img)
            self.labels.append(label)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label
    
class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class VAE(nn.Module):
    def __init__(self, latent_size):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            Conv2dSame(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            Conv2dSame(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            Conv2dSame(128, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*(img_dim//8)*(img_dim//8), latent_size * 2)  # 输出均值和方差
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128*(img_dim//8)*(img_dim//8)),
            nn.ReLU(),
            nn.Unflatten(1, (128, img_dim//8, img_dim//8)),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        latent_params = self.encoder(x)
        mu, logvar = torch.chunk(latent_params, 2, dim=1)
        z = self.reparameterize(mu, logvar)

        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

img_dim = 128
latent_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((img_dim, img_dim)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()])

def custom_collate(batch):
    inputs = [item[0] for item in batch]
    return default_collate(inputs)

model = VAE(latent_size=latent_size).to(device)
model.load_state_dict(torch.load('results/SelfOracle/VOC_VAE.pth'))

test_set = datasets.VOCDetection('data/VOC', year="2007", image_set='test')

criterion = nn.MSELoss(reduction='mean')
batch_size = 1
model.eval()
model.to(device)

chunk_size = 50
filename = 'results/SelfOracle/VOC_validity.npy'
result_selfOracle = {}
num = 0
if os.path.exists(filename):
    existed = np.load(filename,allow_pickle=True).item()
else:
    existed = {}
for i in range(len(mrs)):
    for cmr in permutations(range(len(mrs)), i+1):
        if cmr in existed.keys():
            continue

        result_selfOracle[cmr] = []
        followup_test_set = CustomDataset(test_set, cmr, transform=transform)
        followup_test_loader = DataLoader(followup_test_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        with torch.no_grad():
            for batch_idx, data in enumerate(followup_test_loader):
                data = data.to(device)
                recon, _, _ = model(data)
                error = criterion(recon, data)
                result_selfOracle[cmr].append(error.cpu().item())
        print(cmr)

        num += 1
        if num == chunk_size:   
            result_selfOracle.update(existed)
            np.save(filename, result_selfOracle)
            existed = result_selfOracle
            num = 0
            result_selfOracle = {}

if len(result_selfOracle) != 0:
    result_selfOracle.update(existed)
    np.save(filename, result_selfOracle)

# check
print(len(np.load(filename, allow_pickle=True).item()))