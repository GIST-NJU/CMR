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
from tqdm import tqdm
import insightface
import numpy as np
from mr_utils import mrs
from dataloaders.utkface import UTKFaceDataset_v3

coco_info =  {
    "data_name": "coco",
    "data": "data/source/COCO",
    "annotation_file": 'data/source/COCO/image_info_test2014.json',
    "phase": "test",
    "num_classes": 80,
}

class FollowupDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]

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
        dataset = torchvision.datasets.MNIST(root='./data/source', train=False, download=False, transform=transform)
    elif dataset_name == 'Caltech256':
        caltech256_dataset = torchvision.datasets.Caltech256(root='data/source', download=False)
        X = [caltech256_dataset[i][0] for i in range(len(caltech256_dataset))]
        y = [caltech256_dataset[i][1] for i in range(len(caltech256_dataset))]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=18, stratify=y)
        dataset = CustomDataset(list(zip(X_test, y_test)), transform=transform)
    elif dataset_name == 'VOC':
        dataset = torchvision.datasets.VOCDetection('data/source/VOC', year="2007", image_set='test', transform=transform)
    elif dataset_name == 'COCO':
        dataset = torchvision.datasets.CocoDetection(os.path.join(coco_info['data'],'{}2014'.format(coco_info['phase'])), 
            annFile=coco_info['annotation_file'], transform=transform)
    elif dataset_name == 'UTKFace':
        test_path = "data/source/UTKFace/UTK_test_selected"
        dataset = UTKFaceDataset_v3(data_path=test_path, augmentation=None, train=False, transform=transform)
    else:
        print('Dataset not supported')
        sys.exit(1)
    return dataset

def train_lenet5():
    save_path = os.path.join('results', 'features','lenet5', 'lenet5_mnist.pth')
    if os.path.exists(save_path):
        return
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST('./data/source', train=True, download=True, transform=transform)
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
        followup_path = os.path.join('data', 'followup', datasetname, str(i))
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
        # print(i)
        save_path = os.path.join('results', 'features', 'vgg16', f'{datasetname}_{i}.pt')
        if os.path.exists(save_path):
            continue
        followup_path = os.path.join('data', 'followup', datasetname, str(i))
        followup_testset = FollowupDataset(followup_path, transform=transform)
        loader_followup = torch.utils.data.DataLoader(followup_testset, batch_size=batch_size, shuffle=False, num_workers=4)
        all_features = []
        with torch.no_grad():
            for images in tqdm(loader_followup, desc=f'Extracting features for mr{i}'):
                images = images.to(device)
                features = model(images)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
                all_features.append(features.cpu())
        all_features = torch.cat(all_features, dim=0)
        torch.save(all_features, save_path)
    print(f"{datasetname} features saved")


def extract_and_save_insightface_features(datasetname, img_size=128):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # convert to cv2.imread format
        lambda img: np.array(img),
        lambda img: img[..., ::-1].copy(),  # convert RGB to BGR
    ])
    model = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    model.prepare(
        ctx_id=0 if torch.cuda.is_available() else -1,
        det_size=(img_size, img_size),
        det_thresh=0.1
    )

    # source
    save_path = os.path.join('results', 'features', 'insightface', f'{datasetname}.pt')
    if not os.path.exists(save_path):
        dataset = load_testset(datasetname, transform=transform)
        all_features = []
        with torch.no_grad():
            for images in tqdm(dataset, desc=f'Extracting features for {datasetname}'):
                img = images['image']
                face = model.get(img=img, max_num=1)[0]
                all_features.append(np.array(face.embedding))
        all_features = torch.Tensor(np.array(all_features))
        torch.save(all_features, save_path)

    # followup
    for i in range(len(mrs)):
        save_path = os.path.join('results', 'features', 'insightface', f'{datasetname}_{i}.pt')
        if os.path.exists(save_path):
            continue
        followup_path = os.path.join('data', 'followup', datasetname, str(i))
        follow_testset = UTKFaceDataset_v3(
            data_path=followup_path,
            augmentation=None,
            train=False,
            transform=transform
        )
        all_features = []
        with torch.no_grad():
            for images in tqdm(follow_testset, desc=f'Extracting features for {datasetname}_{i}'):
                img = images['image']
                face = model.get(img=img, max_num=1)[0]
                all_features.append(np.array(face.embedding))
        all_features = torch.Tensor(np.array(all_features))
        torch.save(all_features, save_path)


def process_dataset(dataset):
    if dataset=='MNIST':
        os.makedirs(os.path.join('results', 'features', 'lenet5'), exist_ok=True)
        train_lenet5()
        extract_and_save_lenet5_features(dataset)
    elif dataset in ['UTKFace']:
        os.makedirs(os.path.join('results', 'features', 'insightface'), exist_ok=True)
        extract_and_save_insightface_features(dataset)
    else:
        os.makedirs(os.path.join('results', 'features', 'vgg16'), exist_ok=True)
        extract_and_save_vgg16_features(dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name, e.g. MNIST')
    return parser.parse_args()

def validate_args(args):
    if args.dataset not in ['ALL', 'MNIST', 'Caltech256', 'VOC', 'COCO', 'UTKFace']:
        print(f"[ERROR] Dataset '{args.dataset}' is not sopported")
        print("Supported datasets: ALL, MNIST, Caltech256, VOC, COCO, UTKFace")
        sys.exit(1)

def main():
    args = parse_args()
    validate_args(args)
    if args.dataset == 'ALL':
        for dataset in ['MNIST', 'Caltech256', 'VOC', 'COCO', 'UTKFace']:
            process_dataset(dataset)
    else:
        process_dataset(args.dataset)

if __name__ == "__main__":
    main()
