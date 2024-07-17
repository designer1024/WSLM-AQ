import numpy as np
import torch
import torch.nn.functional as F

def iou_score(array1, array2):
    if torch.is_tensor(array1):
        array1 = array1.data.cpu().numpy()
    if torch.is_tensor(array2):
        array2 = array2.data.cpu().numpy()
    array1 = array1.flatten()
    array2 = array2.flatten()

    intersection = np.logical_and(array1, array2).sum()
    union = np.logical_or(array1, array2).sum()

    iou = intersection / union if union != 0 else 0

    return iou


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
