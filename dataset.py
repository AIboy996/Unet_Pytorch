"""cook dataset for train"""
from torch.utils.data import Dataset
import numpy as np
import os
import re
# for data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def factorize(arr):
            res = np.zeros(arr.shape)
            res[arr==0] = 0     # Others
            res[arr==200] = 1   # Myocardium
            res[arr==500] = 2   # RightVentricle
            res[arr==600] = 3   # LeftVentricle
            return res

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        mask = self.factorize(np.load(self.mask_paths[idx]))
        position = re.match('.*?slice(?P<position>\d+)', self.mask_paths[idx]).groupdict()['position']
        position = int(position)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask, position

# data augmentation
DATA_TRANSFORM = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.2),
    ToTensorV2(),
], is_check_shapes=False)

def load_dataset(data_path, image_suffix='_img.npy', mask_suffix='_lab.npy'):
    image_path_list = []
    mask_path_list = []
    for file in os.listdir(data_path):
        if file.startswith('.'):
            continue
        elif file.endswith(image_suffix):
            image_path_list.append(os.path.join(data_path, file))
        elif file.endswith(mask_suffix):
            mask_path_list.append(os.path.join(data_path, file))
        else:
            print(file)
            raise Exception('Dataset Error')
    print(f'loaded {len(image_path_list) = }, {len(mask_path_list) = } from {data_path}')

    # data set
    dataset = CustomDataset(image_paths=image_path_list,
                            mask_paths=mask_path_list,
                            transform=DATA_TRANSFORM)
    return dataset