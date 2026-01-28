import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ID_MAPPING = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

CLASSES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", 
    "Ground Clutter", "Flowers", "Logs", "Rocks", 
    "Landscape", "Sky"
]

class DualityDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.ids = [f for f in os.listdir(images_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __getitem__(self, i):
        img_name = self.ids[i]
        image = cv2.imread(os.path.join(self.images_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255.0
        image = image.transpose(2, 0, 1)

        if self.masks_dir:
            mask_name = img_name.replace('.jpg', '.png')
            mask = cv2.imread(os.path.join(self.masks_dir, mask_name), 0)
            mask_mapped = np.zeros_like(mask)
            for raw_id, train_id in ID_MAPPING.items():
                mask_mapped[mask == raw_id] = train_id
            return torch.from_numpy(image), torch.from_numpy(mask_mapped).long()
        
        return torch.from_numpy(image), img_name

    def __len__(self):
        return len(self.ids)
