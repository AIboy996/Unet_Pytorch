import matplotlib.pyplot as plt
import numpy as np

# Define a function to visualize images, masks, and predictions
def visualize_images_with_masks(title, images, masks_true, masks_pred, num_rows=5):
    fig, axes = plt.subplots(num_rows, 3, figsize=(10, 15))
    for i in range(num_rows):
        ax = axes[i]
        idx = np.random.randint(len(images))
        
        ax[0].imshow(images[idx][:], cmap='gray')
        ax[0].set_title(f'Image{i+1}')
        ax[0].axis('off')
        
        ax[1].imshow(masks_true[idx], cmap='gray')
        ax[1].set_title('Ground Truth Mask')
        ax[1].axis('off')
        
        ax[2].imshow(masks_pred[idx], cmap='gray')
        ax[2].set_title('Predicted Mask')
        ax[2].axis('off')
    fig.savefig(fname=f'./fig/{str(title)}.jpg', bbox_inches='tight')
