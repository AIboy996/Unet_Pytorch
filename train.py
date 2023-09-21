"""train the model"""
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# for data process
import numpy as np
# local library
from Net import UNet, FNN
from plot import visualize_images_with_masks, visualize_train_process
from dataset import load_dataset
from dicescore import dice_loss
from evaluate import evaluate

# hyperparameter
n_epoch = 1
batch_size = 8
learning_rate = 0.001
alpha, beta = 0.2,0.6 # alpha for Cross Entropy, beta for Dice Loss, 1-alpha-beta for MSE of FNN

# constants
N_CLASSES  = 4 # LV, RV, Myo and others
N_CHANNELS = 1 # single channel
UNET_PATH = './checkpoints/model_unet.pth'
BEST_MODEL_PATH = './checkpoints/best_model_unet.pth'
FNN_PATH = './checkpoints/model_fnn.pth'

# load data
train_set = load_dataset('../dataset/train_data_npy/')
val_set = load_dataset('../dataset/val_data_npy/')

train_dataloader = DataLoader(train_set,
                        batch_size=batch_size,
                        shuffle=True)
val_dataloader = DataLoader(val_set,
                        batch_size=batch_size,
                        shuffle=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print('train on ', device.type)

# Unet model
try:
    model_unet = torch.load(UNET_PATH)
    print('loaded unet model from ', UNET_PATH)
except Exception as e:
    print('load unet model failed', repr(e))
    model_unet = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
criterion_unet = nn.CrossEntropyLoss()
optimizer_unet = optim.Adam(model_unet.parameters(), lr=learning_rate)

# FNN
input_dim = 256*(256//16)*(256//16)
"""
Explanation: input dimension of FNN
256 = num of channels
256 = width/height of input image
16 = 2**4 where 4 = times of downscaling in UNet
"""
hidden_dim1 = 256
hidden_dim2 = 64
output_dim = 1

try:
    model_fnn = torch.load(FNN_PATH)
    print('loaded fnn model from ', FNN_PATH)
except Exception as e:
    print('load fnn model failed', repr(e))
    model_fnn = FNN(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)
criterion_fnn = nn.MSELoss() # nearly the same as Frobenius norm
optimizer_fnn = optim.Adam(model_fnn.parameters(), lr=learning_rate)

# train loop
train_log = [
    # (epoch, train_loss, val_score)
]
best_score = 0
for epoch in range(n_epoch):
    model_unet.train()
    model_fnn.train()
    train_loss = 0.0

    for batch_idx, (img, msk, position) in enumerate(train_dataloader):
        # move tensor to device(cuda or mps)
        img = img.to(torch.float32).to(device)
        msk = msk.to(torch.int64).to(device)
        position = position.to(torch.int64).to(device)
        
        # turn msk into one hot tensor
        msk = nn.functional.one_hot(msk, N_CLASSES)
        msk = torch.permute(msk, (0,3,1,2)) # shape = (batch_size, n_channels, width, height)
        
        # turn into float
        msk = msk.float()
        position = position.float()
        img = img.float()
        
        # calcute loss of UNet
        optimizer_unet.zero_grad()
        outputs, input_for_FNN = model_unet(img)
        loss_cross = criterion_unet(outputs, msk)
        loss_dice = dice_loss(outputs, msk, multiclass=True)
        
        # calcute loss of FNN
        optimizer_fnn.zero_grad()
        input_for_FNN = input_for_FNN.flatten(start_dim=1)
        position_pred = model_fnn(input_for_FNN)
        loss_mse = criterion_fnn(position_pred.flatten(), position)

        # total loss
        
        loss = alpha*loss_cross + beta*loss_dice + (1-alpha-beta)*loss_mse
        loss.backward()
        optimizer_fnn.step()
        optimizer_unet.step()
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
    val_score = evaluate(model_unet, val_dataloader, device).cpu()
    if val_score > best_score:
        best_score = val_score
        torch.save(model_unet, BEST_MODEL_PATH)
    train_log.append((epoch+1, train_loss, val_score))
    print(f"Epoch [{epoch+1}/{n_epoch}] Train Loss: {train_loss:.4f}, Validation Score: {val_score:.4f}")

visualize_train_process(train_log)
torch.save(model_unet, UNET_PATH)
torch.save(model_fnn, FNN_PATH)