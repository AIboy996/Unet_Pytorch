# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# for data process
import os
import numpy as np
# local library
from Net import UNet
from plot import visualize_images_with_masks
from dataset import load_dataset
from dicescore import dice_loss
from evaluate import evaluate

# get the path of image and mask
train_set = load_dataset('../dataset/train_data_npy/')
val_set = load_dataset('../dataset/val_data_npy/')
# test_set = load_dataset('../dataset/test_data_npy/')

batch_size = 8
train_dataloader = DataLoader(train_set,
                        batch_size=batch_size,
                        shuffle=True)
val_dataloader = DataLoader(val_set,
                        batch_size=batch_size,
                        shuffle=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print('train on ', device.type)

# hyperparameter
n_epoch = 2
learning_rate = 0.001
early_stopping_patience = 10

# Unet model
n_classes  = 4 # LV, RV, Myo and others
n_channels = 1 # single channel
model = UNet(n_channels=n_channels, n_classes=n_classes).to(device)
# model = torch.load('./checkpoints/model.pth')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train loop
for epoch in range(n_epoch):
    model.train()
    train_loss = 0.0

    for batch_idx, (img, msk) in enumerate(train_dataloader):
        img, msk = img.to(torch.float32).to(device), msk.to(torch.int64).to(device)
        img = img.float()
        msk = torch.nn.functional.one_hot(msk, n_classes)
        msk = torch.permute(msk, (0,3,1,2)) # shape = (batch_size, n_channels, width, height)
        msk = msk.float()

        optimizer.zero_grad()
        outputs = model(img)
        # loss
        loss = criterion(outputs, msk)
        loss += dice_loss(outputs, msk, multiclass=True)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        # plot every 500 batch
        if batch_idx%500 == 0:
            
            image = img.cpu().numpy().transpose(0, 2, 3, 1)
            predicted_mask = outputs.cpu().detach().numpy()
            ground_truth_mask = msk.cpu().detach().numpy()
            predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
            predicted_mask = np.argmax(predicted_mask, axis=1)
            ground_truth_mask = np.argmax(ground_truth_mask, axis=1)
            visualize_images_with_masks(epoch+1, image, ground_truth_mask, predicted_mask, num_rows=5)
    train_loss /= len(train_dataloader)
    val_score = evaluate(model, val_dataloader, device)
    
    print(f"Epoch [{epoch+1}/{n_epoch}] Train Loss: {train_loss:.4f}, Validatin Score: {val_score:.4f}")

torch.save(model, './checkpoints/model.pth')