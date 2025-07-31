#!/usr/bin/env python3
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from mr_utils import mrs

coco_info =  {
    "data_name": "coco",
    "data": "data/COCO",
    "annotation_file": 'data/COCO/image_info_test2014.json',
    "phase": "test",
    "num_classes": 80,
}

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

class LeNet5(nn.Module):
    def __init__(self, extract_features=False):
        super(LeNet5, self).__init__()
        self.extract_features = extract_features
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.avg_pool2d(F.tanh(self.conv1(x)), 2)
        x = F.avg_pool2d(F.tanh(self.conv2(x)), 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.tanh(self.fc1(x))
        if self.extract_features:
            return F.tanh(self.fc2(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

def load_testset(dataset_name, transform = None):
    if dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    elif dataset_name == 'Caltech256':
        caltech256_dataset = torchvision.datasets.Caltech256(root='data', download=False)
        X = [caltech256_dataset[i][0] for i in range(len(caltech256_dataset))]
        y = [caltech256_dataset[i][1] for i in range(len(caltech256_dataset))]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=18, stratify=y)
        dataset = CustomDataset(list(zip(X_test, y_test)), transform=transform)
    elif dataset_name == 'VOC':
        dataset = torchvision.datasets.VOCDetection('data/VOC', year="2007", image_set='test', transform=transform)
    else: #'COOC'
        dataset = torchvision.datasets.CocoDetection(os.path.join(coco_info['data'],'{}2014'.format(coco_info['phase'])), 
            annFile=coco_info['annotation_file'], transform=transform)
    return dataset

def train_lenet5():
    save_path = os.path.join('results', 'features','lenet5', 'lenet5_mnist.pth')
    if os.path.exists(save_path):
        return
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model = LeNet5()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(20):
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    torch.save(model.state_dict(), save_path)

def extract_and_save_lenet5_features(datasetname):
    model = LeNet5(extract_features=True)
    save_path = os.path.join('results', 'features','lenet5', 'lenet5_mnist.pth')
    model.load_state_dict(torch.load(save_path))
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # source
    save_path = os.path.join('results', 'features', 'lenet5', f'{datasetname}.pt')
    if not os.path.exists(save_path):
        dataset = load_testset(datasetname, transform=transform)
        test_loader = DataLoader(dataset, batch_size=1000, shuffle=False)
        loader = test_loader
        all_features = []
        with torch.no_grad():
            for images, _ in loader:
                features = model(images)
                all_features.append(features)
        all_features = torch.cat(all_features, dim=0)
        torch.save(all_features, save_path)

    # followup
    for i in range(len(mrs)):
        save_path = os.path.join('results', 'features', 'lenet5', f'{datasetname}_{i}.pt')
        if os.path.exists(save_path):
            continue
        followup_path = os.path.join('followup', datasetname, str(i))
        followup_testset = FollowupDataset(followup_path, transform=transform)
        loader_followup = torch.utils.data.DataLoader(followup_testset, batch_size=1000, shuffle=False)
        all_features = []
        with torch.no_grad():
            for images in loader_followup:
                features = model(images)
                all_features.append(features)
        all_features = torch.cat(all_features, dim=0)
        torch.save(all_features, save_path)
    print(f"{datasetname} features saved")

def collate_fn_empty_labels(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [[] for _ in batch]
    return images, labels

def extract_and_save_vgg16_features(datasetname, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = load_testset(datasetname, transform=transform)
    weights = torchvision.models.VGG16_Weights.DEFAULT
    model = torchvision.models.vgg16(weights=weights).features.to(device)
    model.eval()

    # source
    save_path = os.path.join('results', 'features', 'vgg16', f'{datasetname}.pt')
    if not os.path.exists(save_path):
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_empty_labels, num_workers=4)
        all_features = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                features = model(images)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
                all_features.append(features.cpu())
        all_features = torch.cat(all_features, dim=0)
        torch.save(all_features, save_path)

    # followup
    for i in range(len(mrs)):
        print(i)
        save_path = os.path.join('results', 'features', 'vgg16', f'{datasetname}_{i}.pt')
        if os.path.exists(save_path):
            continue
        followup_path = os.path.join('followup', datasetname, str(i))
        followup_testset = FollowupDataset(followup_path, transform=transform)
        loader_followup = torch.utils.data.DataLoader(followup_testset, batch_size=batch_size, shuffle=False, num_workers=4)
        all_features = []
        with torch.no_grad():
            for images in loader_followup:
                images = images.to(device)
                features = model(images)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
                all_features.append(features.cpu())
        all_features = torch.cat(all_features, dim=0)
        torch.save(all_features, save_path)
    print(f"{datasetname} features saved")

def main():
    for dataset in ['MNIST', 'caltech256', 'VOC', 'COCO']:
        if dataset=='MNIST':
            os.makedirs(os.path.join('results', 'features', 'lenet5'), exist_ok=True)
            train_lenet5()
            extract_and_save_lenet5_features(dataset)
        else:
            os.makedirs(os.path.join('results', 'features', 'vgg16'), exist_ok=True)
            extract_and_save_vgg16_features(dataset)

if __name__ == "__main__":
    main()