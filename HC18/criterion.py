from torch import nn
import torch
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim
from HC18.model import UNet
model = UNet()
class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.
    
    Args:
        weight: A float tensor for class weighting (optional).
    """

    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-5
        if weight is not None:
            self.weight = torch.tensor(weight, dtype=torch.float32)
        else:
            self.weight = None

    def forward(self, predict, target):
        """
        Args:
            predict: Tensor of shape (N, 1, H, W) with raw logits.
            target: Tensor of shape (N, H, W) with binary values {0,1}.
        Returns:
            Dice loss scalar.
        """
        predict = torch.sigmoid(predict)  # Convert logits to probabilities

        predict = predict.view(predict.shape[0], -1)  # Flatten to (N, H*W)
        target = target.view(target.shape[0], -1)  # Flatten to (N, H*W)

        # Compute Dice coefficient
        intersection = (predict * target).sum(dim=1)  # Sum over spatial dimensions
        union = predict.sum(dim=1) + target.sum(dim=1)  # Sum of both

        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # Per-batch Dice

        # Apply class weight if provided
        dice_loss = 1 - dice_coef.mean()  # Mean loss over batch
        if self.weight is not None:
            self.weight = self.weight.to(predict.device, dtype=predict.dtype)
            dice_loss *= self.weight  # Apply weight scaling

        return dice_loss
    
def iou_coefficient(predict, target, threshold=0.5, smooth=1e-5):
    """
    Computes the Intersection over Union (IoU) for binary segmentation.
    
    Args:
        predict: Tensor of shape (N, 1, H, W) with raw logits or probabilities.
        target: Tensor of shape (N, H, W) with binary values {0,1}.
        threshold: Threshold to binarize predictions (default=0.5).
        smooth: Small value to avoid division by zero.
    
    Returns:
        IoU score (scalar value).
    """
    predict = torch.sigmoid(predict)  # Convert logits to probabilities
    predict = (predict > threshold).float()  # Binarize predictions

    predict = predict.view(predict.shape[0], -1)  # Flatten to (N, H*W)
    target = target.view(target.shape[0], -1)  # Flatten to (N, H*W)

    intersection = (predict * target).sum(dim=1)  # Element-wise multiplication and sum
    union = (predict + target).sum(dim=1) - intersection  # Union = A + B - Intersection

    iou = (intersection + smooth) / (union + smooth)  # Compute IoU per batch

    return iou.mean()  # Average over batch

optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True)