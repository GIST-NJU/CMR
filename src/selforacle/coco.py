import numpy as np
from scipy.stats import gamma
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch.utils.data import ConcatDataset

class FollowupDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


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
            nn.Linear(128*(img_dim//8)*(img_dim//8), latent_size * 2)
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


def train_vae():
    save_path = 'results/validity/COCO_VAE.pth'
    if os.path.exists(save_path):
        model.load_state_dict(torch.load('results/validity/COCO_VAE.pth'))
        return

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            MSE = criterion(recon_batch, data)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = MSE + KLD
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader.dataset)}")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += criterion(recon_batch, data).item()

    test_loss /= len(test_loader.dataset)
    print(f"Test set loss: {test_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


def calculate_threshold():
    save_path = 'results/validity/COCO_threshold.txt'
    if os.path.exists(save_path):
        return
    
    criterion = nn.MSELoss(reduction='none')
    error_testing = []
    model.load_state_dict(torch.load('results/validity/COCO_VAE.pth'))
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, _, _ = model(data)
            batch_error = criterion(recon_batch, data)
            error_testing.extend([torch.mean(batch_error[i]).cpu().item() for i in range(batch_error.shape[0])])
            #print(len(error_testing))
            #print(error_testing)
    #print(error_testing)
    shape, loc, scale = gamma.fit(error_testing, floc=0)
    false_alarm = 0.0001
    threshold = gamma.ppf(1-false_alarm, shape, loc, scale)
    print(threshold)
    print(np.where(error_testing>threshold)[0].size)

    with open(save_path, 'w') as f:
        f.write(f'False_alarm: {false_alarm}\n')
        f.write(f'Threshold: {threshold}\n')

def predict_validity():
    criterion = nn.MSELoss(reduction='none')
    result_selfOracle = {}
    batch_size = 4096
    model.eval()
    model.to(device)

    followup_dir = 'data/followup/COCO'
    entries = os.listdir(followup_dir)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(followup_dir, entry))]
    folders = sorted(folders)
    for folder in folders:
        cmr = tuple(int(char) for char in folder)
        result_selfOracle[cmr] = []
        followup_path = os.path.join(followup_dir, folder)
        followup_test_set = FollowupDataset(followup_path, transform=transform)
        followup_test_loader = DataLoader(followup_test_set, batch_size=batch_size, shuffle=False, num_workers=2)
        
        with torch.no_grad():
            for batch_idx, data in enumerate(followup_test_loader):
                data = data.to(device)
                recon, _, _ = model(data)
                error = criterion(recon, data)
                result_selfOracle[cmr].extend([torch.mean(error[i]).cpu().item() for i in range(error.shape[0])])
        print(cmr)
    np.save('results/validity/COCO_validity.npy', result_selfOracle)

def run():
    train_vae()
    calculate_threshold()
    predict_validity()

img_dim = 128
epochs = 50
batch_size = 2048
lr = 1e-3
input_size = 3 * 128 * 128
latent_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((img_dim, img_dim)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()])

def custom_collate(batch):
    inputs = [item[0] for item in batch]
    return default_collate(inputs)

data_dir = 'data/source/COCO/'
coco_train = datasets.CocoDetection(root=data_dir + 'train2014',
                            annFile=data_dir + 'instances_train2014.json',
                            transform=transform)
coco_val = datasets.CocoDetection(root=data_dir + 'val2014',
                          annFile=data_dir + 'instances_val2014.json',
                          transform=transform)
train_set = ConcatDataset([coco_train, coco_val])
print(len(train_set))
test_set = datasets.CocoDetection(root=data_dir + 'test2014',
                            annFile=data_dir + 'image_info_test2014.json',
                            transform=transform)
print(len(test_set))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=8)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=8)
model = VAE(latent_size=latent_size).to(device)