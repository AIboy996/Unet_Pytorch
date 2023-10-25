"""function for plot"""
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import io
from PIL import Image

TIME = time.strftime(r"%m-%d(%I.%M%p)")

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
    # if not os.path.exists(f'./fig/train_at_{TIME}'):
    #     os.mkdir(f'./fig/train_at_{TIME}')
    bio = io.BytesIO()
    fig.savefig(bio, bbox_inches='tight', dpi=100)
    return np.array(Image.open(bio)).transpose((2,0,1))