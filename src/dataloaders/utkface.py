from torch.utils.data import Dataset
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import math


def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)


class NumpyToTensorRawTransform:
    def __call__(self, img):
        return torch.tensor(np.asarray(img))

    def __repr__(self) -> str:
        return "torch.tensor(np.asarray(x))"


class UTKFaceDataset_v3(Dataset):
    def __init__(self, data_path, augmentation, train, transform=None, selected_indices=None, **kwargs):

        self.data_path = data_path
        self.train = train

        self.image_paths = []
        self.labels = []

        self.class_labels = []

        if not self.train and augmentation:
            augmentation.type="attribute_test_transform"

        self.transform = transform

        self.split = "train" if self.train else "test"

        if selected_indices is None or len(selected_indices) == 0:
            for filename in sorted(os.listdir(self.data_path)):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(self.data_path, filename)
                    try:
                        age = int(filename.split("_")[0])
                    except:
                        age = -1 # Unknown

                    self.image_paths.append(image_path)
                    self.labels.append(age)
        else:
            for idx in selected_indices:
                filename = f"{idx:05d}.png"
                image_path = os.path.join(self.data_path, filename)
                self.image_paths.append(image_path)
                self.labels.append(-1)

        labels_save_path = os.path.join('data', 'source', 'UTKFace', f'utk_{self.split}_labels.npy')
        if not os.path.exists(labels_save_path):
            np.save(labels_save_path, np.array(self.labels))
        if self.labels[0] == -1:
            self.labels = np.load(labels_save_path).tolist()
            if selected_indices is not None and len(selected_indices) > 0:
                self.labels = [self.labels[idx] for idx in selected_indices]

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        age = self.labels[idx]
        
        dis = [normal_sampling(age, i) for i in range(101)]
        dis = torch.Tensor(dis)
        dis = F.normalize(dis, p=1, dim=0)     

        return {'image':img, 'label':{"avg_label":age, "distribution":dis}, 'filename':img_path}

    def __len__(self):
        return len(self.image_paths)
    
    def __repr__(self):
        return self.__class__.__name__ + \
               f'\nsplit: {self.split}\ndataset_len: {len(self.image_paths)}\naugmentation: {self.transform}'
