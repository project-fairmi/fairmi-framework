import torch
import torch.nn as nn
from fairret.statistic import PositiveRate
from fairret.loss import NormLoss

class FairCrossEntropyLoss(nn.Module):
    """
    Combines CrossEntropyLoss with NormLoss for fairness.
    Accepts all parameters of CrossEntropyLoss via kwargs.
    """
    
    def __init__(self, fairness_weight: float = 1.0, **kwargs):
        """
        Args:
            fairness_weight: Weight for the fairness loss
            **kwargs: All arguments accepted by nn.CrossEntropyLoss
                     (weight, size_average, ignore_index, reduce, reduction, label_smoothing)
        """
        super(FairCrossEntropyLoss, self).__init__()
        
        self.fairness_weight = fairness_weight
        
        self.ce_loss = nn.CrossEntropyLoss(**kwargs)
        
        statistic = PositiveRate()
        self.fairness_loss = NormLoss(statistic)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor, sensitive_attr: torch.Tensor):
        """
        Args:
            input: Model logits (batch_size, num_classes)
            target: True labels (batch_size,)
            sensitive_attr: Sensitive attributes (batch_size,)
        
        Returns:
            torch.Tensor: Combined loss
        """
        ce_loss = self.ce_loss(input, target)
        
        sensitive_attr = sensitive_attr.unsqueeze(-1)
        fairness_loss = self.fairness_loss(input, sensitive_attr)
    
        total_loss = ce_loss + self.fairness_weight * fairness_loss
        
        return total_loss
