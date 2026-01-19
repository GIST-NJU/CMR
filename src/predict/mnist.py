import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import pickle


class FollowupDataset(Dataset):
    def __init__(self, root_dir, transform=None, selected_indices=None):
        self.root_dir = root_dir
        self.transform = transform
        if selected_indices is None or len(selected_indices) == 0:
            self.images = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]
        else:
            self.images = [os.path.join(root_dir, f"{idx:05d}.png") for idx in selected_indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image


class Subset(torch.utils.data.Subset):
    _local_attrs = {'dataset', 'indices'}

    def __getattr__(self, name):
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(
            f"{type(self).__name__} has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        if name in self._local_attrs:
            super().__setattr__(name, value)
        elif hasattr(self.dataset, name):
            setattr(self.dataset, name, value)
        else:
            raise AttributeError(
                f"Cannot set unknown attribute '{name}' "
                f"on {type(self).__name__}"
            )


def test_source(source_num=None):
    save_path = os.path.join(folder_path, model_name+'_source'+(f'_{source_num}' if source_num else '')+'.npy')
    if os.path.exists(save_path):
        print("Source predictions already exist.")
        return
    mnist_testset = datasets.MNIST(root='./data/source', train=False, download=True, transform=transform)
    if source_num:
        with open(f'results/samples/MNIST_{source_num}.pkl', 'rb') as f:
            selected_indices = pickle.load(f)
        mnist_testset = Subset(mnist_testset, selected_indices)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)
    pred_source = np.zeros(len(mnist_testset),dtype=int)
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(test_loader, desc="Predicting source inputs")):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, pred = torch.max(outputs, 1)
            pred_source[i*batch_size:i*batch_size+X.size(0)] = pred.cpu()
    np.save(save_path, pred_source)


def test_followup(augment, cmr_num=None, source_num=None):
    save_path = os.path.join(folder_path, model_name+'_followup'+(f'_{cmr_num}' if cmr_num else '')+(f'_{source_num}' if source_num else '')+'.npy')
    if os.path.exists(save_path):
        print("Followup predictions already exist.")
        return
    pred_followup = {}
    followup_dir = 'data/followup/MNIST'
    if not augment:
        entries = os.listdir(followup_dir)
    else:
        with open(f'results/samples/MNIST_AlexNet_9938_cmr{cmr_num}.pkl', 'rb') as f:
            selected_cmrs = pickle.load(f)
        entries = [''.join(map(str, cmr)) for cmrs in selected_cmrs.values() for cmr in cmrs]
    folders = [entry for entry in entries if os.path.isdir(os.path.join(followup_dir, entry))]
    folders = sorted(folders)
    if source_num:
        with open(f'results/samples/MNIST_{source_num}.pkl', 'rb') as f:
            selected_indices = pickle.load(f)
    else:
        selected_indices = None
    for folder in tqdm(folders, desc="Predicting followup inputs"):
        cmr = tuple(int(char) for char in folder)
        followup_path = os.path.join(followup_dir, folder)
        followup_test_set = FollowupDataset(followup_path, transform=transform, selected_indices=selected_indices)
        temp = np.zeros(len(followup_test_set), dtype=int)
        followup_test_loader = DataLoader(followup_test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        with torch.no_grad():
            for i, X in enumerate(tqdm(followup_test_loader, desc=f"Predicting followup for CMR {cmr}", leave=False)):
                X = X.to(device)
                outputs = model(X)
                _, pred = torch.max(outputs, 1)
                temp[i*batch_size:i*batch_size+X.size(0)] = pred.cpu()
        pred_followup[cmr] = temp
    np.save(save_path, pred_followup)


def load_model(augment):
    global model, model_name, folder_path
    if not augment:
        model_name = 'MNIST_AlexNet_9938'
    elif augment == 'offline':
        model_name = 'MNIST_AlexNet_Aug_offline_9947'
    else:
        model_name = 'MNIST_AlexNet_Aug_online_9938'
    folder_path = './results/predictions/MNIST'
    model = models.alexnet()
    model.classifier[6] = nn.Linear(4096, 10)
    model.load_state_dict(torch.load('./models/'+model_name+'.pth'))
    model.eval()
    model.to(device)
    os.makedirs(folder_path, exist_ok=True)


def run(followup, augment, cmr_num=None, source_num=None):
    load_model(augment)
    if not followup:
        test_source(source_num)
    else:
        test_followup(augment, cmr_num, source_num)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
batch_size = 4096
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
