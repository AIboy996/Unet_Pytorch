"""definition of Dice score"""
import torch
from torch import Tensor
from torch.nn.functional import one_hot

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    # Note that input, target are 0-1 array
    inter = 2 * (input * target).sum(dim=sum_dim) # sum on input==1 and target==1
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def mix_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    res = [dice_coeff(input[:,i,...], target[:,i,...], reduce_batch_first, epsilon) for i in range(input.size(1))]
    return Tensor(res)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    input = torch.permute((one_hot(input.argmax(axis=1))), (0, 3, 1, 2))
    target = torch.permute((one_hot(target.argmax(axis=1))), (0, 3, 1, 2))
    if multiclass:
        return 1 - multiclass_dice_coeff(input, target, reduce_batch_first=True).mean()
    else:
        return 1 - dice_coeff(input, target, reduce_batch_first=True)
