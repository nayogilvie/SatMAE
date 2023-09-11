#Focal Loss for images
#Adding nn functionality from entropy loss
# pylint: disable=arguments-differ

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input_, target, eval=False):

        cross_entropy = -super().forward(input_, target)

        input_prob_argmax = torch.exp(cross_entropy)
        loss = -torch.pow(1.0 - input_prob_argmax, self.gamma) * cross_entropy

        return torch.mean(loss)
        # return torch.sum(loss)

        


