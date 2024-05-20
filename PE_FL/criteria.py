import torch
import torch.nn as nn

loss_names = ['l1', 'l2']

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
    

class HuberLoss(nn.Module):
    def __init__(self, delta=17):  # 290开根号
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        error = target - pred
        valid_mask = (target > 0).detach()
        error = error[valid_mask]
        is_small_error = torch.abs(error) <= self.delta
        small_error_loss = 0.5 * error**2
        large_error_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        loss = torch.where(is_small_error, small_error_loss, large_error_loss)
        return loss.mean()
    

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = pred - target
        loss = torch.log(torch.cosh(diff))
        return loss.mean()