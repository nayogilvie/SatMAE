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
        #print("Eval: ", eval)
        #print("input_ size: ", input_.size())
        cross_entropy = super().forward(input_, target)
        #print("cross_size: ", cross_entropy.size())
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        #print("Targets2: ", target.size())
        #print((target.unsqueeze(1)).size())
        #print((target.unsqueeze(1)).dtype)
        #print((target.unsqueeze(0)).to(torch.int64).size())
        #print(target.to(torch.int64).size())
        #print(F.softmax(input_, 1).size())
        #print(input_.data())
        #Eval doesn't have 5 percentages it is just the output so doesn't need to be paired down
        if (eval == False):
            input_prob = torch.gather(F.softmax(input_, 1), 1, (target).to(torch.int64))
            input_prob_argmax = torch.argmax(input_prob, dim=1)
            #print(torch.pow(1 - input_prob, self.gamma).size())
        else:
            input_prob_argmax = input_
            input_prob_argmax = torch.argmax(input_, dim=1)
        #input_prob = torch.argmax(F.softmax(input_, 1))
        #print("Past here")
        #print(cross_entropy.size())
        #print(input_prob_argmax.size())
        #print("Gama: ", self.gamma, " alpha: ", self.alpha)
        loss = torch.pow(1 - input_prob_argmax, self.gamma) * cross_entropy
        #print("Loss: ", loss.size())
        # return torch.mean(loss)
        return torch.mean(cross_entropy)
        #if self.reduction == 'mean':
        #    return torch.mean(loss)
        #elif (self.reduction == 'sum'):
        #    return torch.sum(loss)
        #else:
        #    return loss 
