import os
import numpy as np
from SimpleITK import GetArrayFromImage, ReadImage

def save2npy(raw_data_dir, save_dir):
    for file in os.listdir(raw_data_dir):
        if file.startswith('.'):
            continue
        suffix = 'lab' if 'lab' in file else 'img'
        itk_img = ReadImage(os.path.join(raw_data_dir, file))
        img_slices = GetArrayFromImage(itk_img)
        for slice in range(img_slices.shape[0]):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.save(os.path.join(save_dir, f'{file[:-11]}_slice{slice+1}_{suffix}'), img_slices[slice])
    delete_empty(save_dir)

def delete_empty(data_dir):
    for file in os.listdir(data_dir):
        if file.startswith('.') or file.endswith('_img.npy'):
            continue
        arr = np.load(os.path.join(data_dir, file))
        if arr.max() == 0: # all balck
            os.remove(os.path.join(data_dir, file))
            os.remove(os.path.join(data_dir, file.replace('lab', 'img')))

if __name__ == "__main__":
    save2npy(raw_data_dir='../dataset/train_original')
    save2npy(raw_data_dir='../dataset/val_data')
    save2npy(raw_data_dir='../dataset/test_data')