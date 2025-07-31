import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

class FollowupDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

def test_source():
    save_path = os.path.join(folder_path, model_name+'_source.npy')
    if os.path.exists(save_path):
        return
    pred_source = np.zeros(len(mnist_testset),dtype=int)
    with torch.no_grad():
        for i,(X,y) in enumerate(test_loader):
            X,y = X.to(device),y.to(device)
            outputs = model(X)
            _, pred = torch.max(outputs, 1)
            pred_source[i*batch_size:i*batch_size+X.size(0)] = pred.cpu()
    np.save(save_path, pred_source)

def test_followup():
    save_path = os.path.join(folder_path, model_name+'_followup.npy')
    if os.path.exists(save_path):
        return
    pred_followup = {}
    followup_dir = 'followup/MNIST'
    entries = os.listdir(followup_dir)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(followup_dir, entry))]
    folders = sorted(folders)
    for folder in folders:
        cmr = tuple(int(char) for char in folder)
        temp = np.zeros(len(mnist_testset), dtype=int)
        followup_path = os.path.join(followup_dir, folder)
        followup_test_set = FollowupDataset(followup_path, transform=transform)
        followup_test_loader = DataLoader(followup_test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        with torch.no_grad():
            for i, X in enumerate(followup_test_loader):
                X = X.to(device)
                outputs = model(X)
                _, pred = torch.max(outputs, 1)
                temp[i*batch_size:i*batch_size+X.size(0)] = pred.cpu()
        pred_followup[cmr] = temp
    np.save(save_path, pred_followup)

def run(followup):
    if not followup:
        test_source()
    else:
        test_followup()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

batch_size = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)
model_name = 'MNIST_AlexNet_9938'
model = models.alexnet()
model.classifier[6] = nn.Linear(4096, 10)
model.load_state_dict(torch.load('./models/'+model_name+'.pth'))
model.eval()
model.to(device)
folder_path = './results/predictions/MNIST'
os.makedirs(folder_path, exist_ok=True)