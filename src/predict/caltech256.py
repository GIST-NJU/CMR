import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.data = []
        for idx, (img, label) in enumerate(dataset):
            self.data.append(img)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, []


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
    caltech256_dataset = datasets.Caltech256(root='data/source', download=True)
    X = [caltech256_dataset[i][0] for i in range(len(caltech256_dataset))]
    y = [caltech256_dataset[i][1] for i in range(len(caltech256_dataset))]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=18, stratify=y)
    test_set = CustomDataset(zip(X_test, y_test), transform=transform)
    if source_num:
        with open(f'results/samples/Caltech256_{source_num}.pkl', 'rb') as f:
            selected_indices = pickle.load(f)
        test_set = Subset(test_set, selected_indices)
    pred_source = np.zeros(len(test_set),dtype=int)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(test_loader, desc="Predicting source inputs")):
            X = X.to(device)
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
    followup_dir = 'data/followup/Caltech256'
    if not augment:
        entries = os.listdir(followup_dir)
    else:
        with open(f'results/samples/Caltech256_DenseNet121_6838_cmr{cmr_num}.pkl', 'rb') as f:
            selected_cmrs = pickle.load(f)
        entries = set([''.join(map(str, cmr)) for cmrs in selected_cmrs.values() for cmr in cmrs])
    folders = [entry for entry in entries if os.path.isdir(os.path.join(followup_dir, entry))]
    folders = sorted(folders)
    if source_num:
        with open(f'results/samples/Caltech256_{source_num}.pkl', 'rb') as f:
            selected_indices = pickle.load(f)
    else:
        selected_indices = None
    for folder in tqdm(folders, desc="Predicting followup inputs"):
        cmr = tuple(int(char) for char in folder)
        followup_path = os.path.join(followup_dir, folder)
        followup_test_set = FollowupDataset(followup_path, transform=transform, selected_indices=selected_indices)
        followup_test_loader = DataLoader(followup_test_set, batch_size=batch_size, shuffle=False, num_workers=8)
        temp = np.zeros(len(followup_test_set), dtype=int)
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
        model_name = 'Caltech256_DenseNet121_6838'
    elif augment == 'offline':
        model_name = 'Caltech256_DenseNet121_offline_xxxx'
    else:
        model_name = 'Caltech256_DenseNet121_Aug_online_7187'
    folder_path = './results/predictions/Caltech256'
    model = models.densenet121(weights='DEFAULT')
    num_classes = 257
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3), # the mode of some images is L not RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
batch_size = 64
