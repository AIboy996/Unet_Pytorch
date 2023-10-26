"""train the model"""
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# for data process
import time
import os
# local library
from Net import UNet, FNN, AutoEncoder
from plot import visualize_images_with_masks
from dataset import load_dataset
from metrics import dice_loss
from evaluate import evaluate, test_model
# tensorboard
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir="./runs" --port=8899
writer = SummaryWriter()


# add on
ACNN_ON = True
FNN_ON = True

# hyperparameter
n_epoch = 150
batch_size = 16
learning_rate = 0.001
lambda_dice = 0.6
alpha = 0.95   # alpha for Unet
beta = 0.005  # beta for ACNN
# 1-alpha-beta for FNN

# constants
N_CLASSES  = 4 # LV, RV, Myo and others
N_CHANNELS = 1 # single channel

# checkpoints path
HASH_CODE = time.time()
CHECKPOINTS_DIR = f'./checkpoints/{HASH_CODE}'
if not os.path.exists(CHECKPOINTS_DIR):
    os.mkdir(CHECKPOINTS_DIR)
UNET_PATH = CHECKPOINTS_DIR + '/model_unet.pth'
BEST_MODEL_PATH = CHECKPOINTS_DIR + '/best_model_unet.pth'
FNN_PATH = CHECKPOINTS_DIR + '/model_fnn.pth'
ACNN_PATH = CHECKPOINTS_DIR + '/model_acnn.pth'

hparam_dict = dict(
    ACNN_ON = ACNN_ON,
    FNN_ON = FNN_ON,
    CHECKPOINTS_DIR = CHECKPOINTS_DIR,
    lambda_dice = lambda_dice,
    n_epoch = n_epoch,
    batch_size = batch_size,
    learning_rate = learning_rate,
    alpha = alpha,
    beta = beta
)

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
    model_unet = torch.load(UNET_PATH, map_location=device)
    print('loaded unet model from ', UNET_PATH)
except Exception as e:
    print('load unet model failed', repr(e))
    model_unet = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
criterion_unet = nn.CrossEntropyLoss()
optimizer_unet = optim.Adam(model_unet.parameters(), lr=learning_rate)

# FNN
if FNN_ON:
    try:
        model_fnn = torch.load(FNN_PATH, map_location=device)
        print('loaded fnn model from ', FNN_PATH)
    except Exception as e:
        print('load fnn model failed', repr(e))
        model_fnn = FNN(
            input_dim=256*(256//16)*(256//16), 
            hidden_dim1=256, 
            hidden_dim2=64, 
            output_dim=1
        ).to(device)
        """
        Explanation: input dimension of FNN
        256 = num of channels
        256 = width/height of input image
        16 = 2**4 where 4 = times of downscaling in UNet
        """
    criterion_fnn = nn.MSELoss() # nearly the same as Frobenius norm
    optimizer_fnn = optim.Adam(model_fnn.parameters(), lr=learning_rate)

# ACNN
if ACNN_ON:
    try:
        model_acnn = torch.load(ACNN_PATH, map_location=device)
        print('loaded acnn model from ', ACNN_PATH)
    except Exception as e:
        print('load acnn model failed', repr(e))
        model_acnn = AutoEncoder(input_size=(256,256), in_channels=1).to(device)
    criterion_acnn = nn.MSELoss()
    optimizer_acnn = optim.Adam(model_acnn.parameters(), lr=learning_rate)

# train loop
best_val_dice_score = 0
for epoch in range(n_epoch):
    model_unet.train()
    if FNN_ON:
        model_fnn.train()
    if ACNN_ON:
        model_acnn.train()
    train_loss = 0.0

    for batch_idx, (img, msk, position) in enumerate(train_dataloader):
        # move tensor to device(cuda or mps)
        img = img.to(torch.float32).to(device)
        msk = msk.to(torch.int64).to(device) # shape=(batch_size, height, width)
        position = position.to(torch.int64).to(device)

        # turn msk into one hot tensor
        msk = nn.functional.one_hot(msk, N_CLASSES) # shape=(batch_size, height, width, channel)
        msk = torch.permute(msk, (0,3,1,2)) # shape=(batch_size, channel, height, width)
        
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
        if FNN_ON:
            optimizer_fnn.zero_grad()
            input_for_FNN = input_for_FNN.flatten(start_dim=1)
            position_pred = model_fnn(input_for_FNN)
            loss_fnn_mse = criterion_fnn(position_pred.flatten(), position)
        else:
            loss_fnn_mse = 0

        # model prediction
        predicted_mask = (outputs>0.5).to(torch.int64).argmax(axis=1, keepdim=True)

        # all black loss
        all_black_loss = 0
        if predicted_mask.sum() == 0:
            all_black_loss += 10
        ground_truth_mask = msk.argmax(axis=1, keepdim=True)

        # calcute loss of ACNN
        if ACNN_ON:
            optimizer_acnn.zero_grad()
            loss_acnn_mse = criterion_acnn(
                model_acnn(predicted_mask.float()), 
                model_acnn(ground_truth_mask.float())
            )
        else:
            loss_acnn_mse = 0

        # total loss
        loss = alpha*((1-lambda_dice)*loss_cross + lambda_dice*loss_dice) \
                + beta*loss_acnn_mse \
                + (1-alpha-beta)*loss_fnn_mse\
                + all_black_loss
        loss.backward()
        optimizer_unet.step()
        if FNN_ON:
            optimizer_fnn.step()
        if ACNN_ON:
            optimizer_acnn.step()
        train_loss += loss.item()

        # plot every 500 batch
        if batch_idx%500 == 0:
            image = img.cpu().numpy().transpose(0, 2, 3, 1)
            predicted_mask = predicted_mask.cpu().detach().numpy()
            ground_truth_mask = ground_truth_mask.cpu().detach().numpy()
            plot = visualize_images_with_masks(image, ground_truth_mask, predicted_mask, num_rows=5)
            writer.add_image('imgs', plot, epoch)

    # calcute mean loss
    train_loss /= len(train_dataloader)

    # validation
    val_dice_score, assd_score, hd_score = evaluate(model_unet, val_dataloader, device)
    # make checkpoints for best model
    if val_dice_score.mean() > best_val_dice_score:
        best_val_dice_score = float(val_dice_score.mean())
        torch.save(model_unet, BEST_MODEL_PATH)
    
    # logger
    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('val_dice_score_mean', val_dice_score.mean(), epoch)
    writer.add_scalar('val_assd_score', assd_score, epoch)
    writer.add_scalar('val_hd_score', hd_score, epoch)
    writer.add_scalars(
        'val_dice_score_by_class',
        dict(
            zip(
                ['Others','Myocardium','RightVentricle','LeftVentricle'], 
                val_dice_score
            )
        ), 
        epoch
    )
    print(f"Epoch [{epoch+1}/{n_epoch}] Train Loss: {train_loss:.4f}, Validation Dice Score: {val_dice_score}")


test_dice_score, test_assd_score, test_hd_score = test_model(BEST_MODEL_PATH)

# logger
writer.add_hparams(
    hparam_dict=hparam_dict, 
    metric_dict=dict(
        test_dice_score_Others = test_dice_score[0],
        test_dice_score_Myocardium = test_dice_score[1],
        test_dice_score_RightVentricle = test_dice_score[2],
        test_dice_score_LeftVentricle = test_dice_score[3],
        test_assd_score = test_assd_score,
        test_hd_score = test_hd_score,
        best_val_dice_score = best_val_dice_score
        )
)

torch.save(model_unet, UNET_PATH)
if FNN_ON:
    torch.save(model_fnn, FNN_PATH)
if ACNN_ON:
    torch.save(model_acnn, ACNN_PATH)