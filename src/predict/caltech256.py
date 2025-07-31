import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, cmr=None, transform=None):
        self.transform = transform
        self.data = []
        for idx, (img, label) in enumerate(dataset):
            if cmr is not None:
                for index in cmr:
                    img = mrs[index](img, paras[index])
            self.data.append(img)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, []


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
    save_path = os.path.join(folder_path, model_name.split('.')[0]+'_source.npy')
    if os.path.exists(save_path):
        return
    pred_source = np.zeros(len(test_set),dtype=int)
    with torch.no_grad():
        for i,(X,y) in enumerate(test_loader):
            X = X.to(device)
            outputs = model(X)
            _, pred = torch.max(outputs, 1)
            pred_source[i*batch_size:i*batch_size+X.size(0)] = pred.cpu()
    np.save(save_path, pred_source)

def test_followup():
    save_path = os.path.join(folder_path, model_name.split('.')[0]+'_followup.npy')
    if os.path.exists(save_path):
        return
    pred_followup = {}
    followup_dir = 'followup/caltech256'
    entries = os.listdir(followup_dir)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(followup_dir, entry))]
    folders = sorted(folders)
    for folder in folders:
        cmr = tuple(int(char) for char in folder)
        temp = np.zeros(len(test_set), dtype=int)
        followup_path = os.path.join(followup_dir, folder)
        followup_test_set = FollowupDataset(followup_path, transform=transform)
        followup_test_loader = DataLoader(followup_test_set, batch_size=batch_size, shuffle=False, num_workers=8)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3), # the mode of some images is L not RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
caltech256_dataset = datasets.Caltech256(root='data', download=True)
X = [caltech256_dataset[i][0] for i in range(len(caltech256_dataset))]
y = [caltech256_dataset[i][1] for i in range(len(caltech256_dataset))]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=18, stratify=y)
test_set = CustomDataset(zip(X_test, y_test), transform=transform)
batch_size = 128
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
model_name = 'Caltech256_DenseNet121_6838'
model = models.densenet121(weights='DEFAULT')
num_classes = 257
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(torch.load('./models/'+model_name+'.pth'))
model.eval()
model.to(device)
folder_path = './predictions/Caltech256'
os.makedirs(folder_path, exist_ok=True)