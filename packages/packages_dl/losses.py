
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass
'---------------------classification---------------------------'
class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=nn.L1Loss(reduce=None)
    def forward(self,input,target):
        return self.loss(input,target)
    
class MseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=nn.MSELoss(reduce=None)   
    def forward(self,input,target):
        return self.loss(input,target)

class BceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=nn.BCELoss()
    def forward(self,input,target):
        return self.loss(input,target)  


class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self,reduction='mean'):
        super(BinaryCrossEntropyWithLogits,self).__init__()
        self.reduction=reduction
    def forward(self, input, target):
        # print(input.shape)
        # print(target.shape)
        input=input.float()
        target=target.float()
        ce = F.binary_cross_entropy_with_logits(input, target,reduction=self.reduction)
        return ce
    
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction, pos_weight=pos_weight)#,ignore_index=255
    def forward(self, input, target):
        input=input.float()
        target=target.float()
        return self.loss(input, target)
    
class MultilabelSoftMarginLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        return F.multilabel_soft_margin_loss(input, target)
    
class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        if target.ndim==4:
            target=torch.squeeze(target)
            target=target.long()
        ce = F.cross_entropy(input, target)
        return ce
    
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=nn.CrossEntropyLoss(reduce=None)
    def forward(self,input,target):
        # print(target.shape)
        if target.dim() == 2:
            target = torch.argmax(target, dim=1)
        # print(target.shape)
        return self.loss(input,target)
        
class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

'-------------------------Segmentation------------------------------'
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.activation_fn = nn.Softmax()

    def forward(self, pred, target):
        pred = self.activation_fn(pred)
        N = target.size()[0]
        class_num = pred.shape[1]
        pred = pred.argmax(1)
        all_dice = 0
        for idx in range(class_num):
            smooth = 1
            p = (pred == idx).int().reshape(-1)
            t = (target == idx).int().reshape(-1)
            union = p.sum() + t.sum()
            overlap = (p * t).sum()
            # print(idx, uion, overlap)
            dice = 2 * overlap / (union + smooth)
            all_dice = all_dice + dice
        diceloss = 1 - all_dice / (N * class_num)
        return diceloss
    
def _iou(pred, target, size_average = True):
    Iand1 = torch.sum(target * pred, dim=[1,2,3])
    Ior1 = torch.sum(target, dim=[1,2,3]) + torch.sum(pred, dim=[1,2,3]) - Iand1
    IoU = 1- (Iand1 / (Ior1 + 1e-8))

    if size_average==True:
        IoU = IoU.mean()
    return IoU

class IouLoss(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IouLoss, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)
    
class LossJoint(nn.Module):
    def __init__(self, loss_fn_list,  weight):
        super(LossJoint, self).__init__()
        self.loss_fn_list = loss_fn_list
        self.weight = weight

    def forward(self, pred, target):
        loss=0
        import inspect
        import sys
        classlist=inspect.getmembers(sys.modules[__name__],inspect.isclass)
        classdict=dict(classlist)
        for i,item in enumerate(self.loss_fn_list):
            loss_class=classdict[item]()
            loss+=self.weight[i]*loss_class(pred,target)
        return loss