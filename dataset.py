from torch.utils.data import Dataset
import numpy as np

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
            res[arr==0] = 0
            res[arr==200] = 1
            res[arr==500] = 2
            res[arr==600] = 3
            return res

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        mask = self.factorize(np.load(self.mask_paths[idx]))
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask

