import torch.utils.data as data
from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np


urls = {'train_img': 'http://images.cocodataset.org/zips/train2014.zip',
        'val_img': 'http://images.cocodataset.org/zips/val2014.zip',
        'test_img': 'http://images.cocodataset.org/zips/test2014.zip',
        'annotations_trainval2014': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        'image_info_test2014': 'http://images.cocodataset.org/annotations/image_info_test2014.zip'}
cat2idx = {
    'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7,
    'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13,
    'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21,
    'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28,
    'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34,
    'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40,
    'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48,
    'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56,
    'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63,
    'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70,
    'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77,
    'hair dryer': 78, 'toothbrush': 79
}
label_transform = {
    0: 4, 1: 47, 2: 24, 3: 46, 4: 34, 5: 35, 6: 21, 7: 59, 8: 13, 9: 1, 10: 14, 11: 8, 12: 73, 13: 39,
    14: 45, 15: 50, 16: 5, 17: 55, 18: 2, 19: 51, 20: 15, 21: 67, 22: 56, 23: 74, 24: 57, 25: 19, 26: 41,
    27: 60, 28: 16, 29: 54, 30: 20, 31: 10, 32: 42, 33: 29, 34: 23, 35: 78, 36: 26, 37: 17, 38: 52, 39: 66,
    40: 33, 41: 43, 42: 63, 43: 68, 44: 3, 45: 64, 46: 49, 47: 69, 48: 12, 49: 0, 50: 53, 51: 58, 52: 72,
    53: 65, 54: 48, 55: 76, 56: 18, 57: 71, 58: 36, 59: 30, 60: 31, 61: 44, 62: 32, 63: 11, 64: 28, 65: 37,
    66: 77, 67: 38, 68: 27, 69: 70, 70: 61, 71: 79, 72: 9, 73: 6, 74: 7, 75: 62, 76: 25, 77: 75, 78: 40, 79: 22
}

def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx

class COCO2014Classification(data.Dataset):
    def __init__(self, root, annotation_file, transform=None, phase='train'):
        self.root = root
        self.phase = phase
        self.img_path = os.path.join(self.root, '{}2014'.format(self.phase))
        self.coco = COCO(annotation_file)
        self.image_ids = self.coco.getImgIds()
        self.transform = transform
        self.cat2idx = cat2idx
        self.num_classes = len(self.cat2idx)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_meta = self.coco.loadImgs(img_id)[0]
        filename = img_meta['file_name']

        img = Image.open(os.path.join(self.img_path, filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # For MT, the target is not used. 
        # For speed up, target is set 0.
        target = np.zeros(self.num_classes, np.float32)
        return filename, img, target

    def get_cat2id(self):
        return self.cat2idx
