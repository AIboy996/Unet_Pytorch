"""function for plot"""
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from torch.utils.data import DataLoader
from dataset import load_dataset
import torch


# Define a function to visualize images, masks, and predictions
def visualize_images_with_masks(images, masks_true, masks_pred, num_rows=5):
    fig, axes = plt.subplots(num_rows, 3, figsize=(10, 15))
    for i in range(num_rows):
        ax = axes[i]
        idx = np.random.randint(len(images))
        
        ax[0].imshow(images[idx][:], cmap='gray')
        ax[0].set_title(f'Image{i+1}')
        ax[0].axis('off')
        
        ax[1].imshow(masks_true[idx,0,...], cmap='gray')
        ax[1].set_title('Ground Truth Mask')
        ax[1].axis('off')
        
        ax[2].imshow(masks_pred[idx,0,...], cmap='gray')
        ax[2].set_title('Predicted Mask')
        ax[2].axis('off')
    bio = io.BytesIO()
    fig.savefig(bio, bbox_inches='tight', dpi=100)
    plt.close()
    return np.array(Image.open(bio)).transpose((2,0,1))

def visualize_diff_models(*model_paths):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    test_set = load_dataset('../dataset/test_data_npy/', augmentation=False)
    batch_size = 4
    dataloader = DataLoader(test_set,
                        batch_size=batch_size,
                        shuffle=True)
    img, msk, _ = next(iter(dataloader))
    img, msk = img.to(device).to(torch.float32), msk.to(device)
    fig, axes = plt.subplots(
        nrows=batch_size, 
        ncols=len(model_paths)+2, 
        # figsize=(batch_size, len(model_paths)+2),
        sharex=True,
        sharey=True
    )
    fig.subplots_adjust(wspace=0, hspace=-.5)
    clear_axis = lambda ax:[
        ax.set_yticklabels([]),
        ax.set_xticklabels([]),
        ax.set_xticks([]),
        ax.set_yticks([])
    ]
    for row in range(batch_size):
        axes[row, 0].imshow(img[row,0,...].cpu(), cmap='jet')
        clear_axis(axes[row, 0])
        axes[row, 0].set_ylabel(f'{row+1} ', rotation=0)
        axes[row, 1].imshow(msk[row].cpu(), cmap='gray')
        clear_axis(axes[row, 1])
        if row==batch_size-1:
            axes[row, 0].set_xlabel('Raw Image')
            axes[row, 1].set_xlabel('Truth')
    for model_no, model_path in enumerate(model_paths):
        model = torch.load(model_path, map_location=device)
        model.eval()
        msk_pred, _ = model(img)
        msk_pred = msk_pred.argmax(axis=1)
        for row in range(batch_size):
            axes[row, model_no+2].imshow(msk_pred[row].cpu(), cmap='gray')
            clear_axis(axes[row, model_no+2])
            if row==batch_size-1:
                axes[row, model_no+2].set_xlabel(model_path.split('/')[2])
    fig.savefig('./fig/a.png', dpi=200, bbox_inches='tight')

if __name__ == "__main__":
    visualize_diff_models(
        './checkpoints/UNet/best_model_unet.pth',
        './checkpoints/SCN/best_model_unet.pth',
        './checkpoints/SRNN/best_model_unet.pth',
        './checkpoints/SRSCN/best_model_unet.pth'
    )
