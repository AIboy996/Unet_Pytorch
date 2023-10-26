"""evaluate model by Dice score"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_dataset
from metrics import multiclass_dice_coeff, dice_coeff
from medpy.metric.binary import assd, hd

@torch.inference_mode()
def evaluate(net, dataloader, device, amp=True):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = None
    assd_score = 0
    hd_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true, _ = batch

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = F.one_hot(mask_true, net.n_classes)
            mask_true = torch.permute(mask_true, (0,3,1,2)) # shape = (batch_size, n_channels, width, height)
            # predict the mask
            mask_pred, _ = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score_batch = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                dice_score = dice_score_batch if dice_score is None else dice_score_batch+dice_score
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score_batch = multiclass_dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                dice_score = dice_score_batch if dice_score is None else dice_score_batch+dice_score
            
            for pred, true in zip(mask_pred, mask_true):
                assd_score += assd(pred.cpu().detach().numpy(), true.cpu().detach().numpy())
                hd_score += hd(pred.cpu().detach().numpy(), true.cpu().detach().numpy())
    net.train()
    return dice_score / max(num_val_batches, 1), assd_score / max(num_val_batches, 1), hd_score / max(num_val_batches, 1)

def test_model(model_path):
    test_set = load_dataset('../dataset/test_data_npy/')
    dataloader = DataLoader(test_set,
                        batch_size=16,
                        shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model_unet = torch.load(model_path, map_location=device)
    dice_score, assd_score, hd_score = evaluate(model_unet, dataloader, device)
    return dice_score, assd_score, hd_score

if __name__ == "__main__":
    print(test_model('./checkpoints(Unet)/best_model_unet.pth'))