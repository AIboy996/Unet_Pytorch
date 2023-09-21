import matplotlib.pyplot as plt
import numpy as np
import time
import os

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
    if not os.path.exists('./fig'):
        os.mkdir('./fig')
    fig.savefig(fname=f'./fig/Epoch{title}.jpg', bbox_inches='tight')

def visualize_train_process(train_log):
    fig, ax = plt.subplots(figsize=(10, 10))
    epoch_l, train_loss, val_score = list(zip(*train_log))
    ax.plot(epoch_l, train_loss, label='train loss')
    ax.plot(epoch_l, val_score, label='validation score')
    ax.set_xticks(epoch_l)
    ax.set_xlabel('epoch')
    ax.set_xlabel('loss/score')
    ax.legend()
    if not os.path.exists('./fig'):
        os.mkdir('./fig')
    fig.savefig(fname=f'./fig/train_at_{time.strftime(r"%m-%d-%I%p")}.jpg', bbox_inches='tight')