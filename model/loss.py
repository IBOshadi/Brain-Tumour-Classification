import torch.nn.functional as F
import torch.nn as nn

def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)

# def cross_entropy_loss(weight=None):
#     return nn.CrossEntropyLoss(weight=weight)