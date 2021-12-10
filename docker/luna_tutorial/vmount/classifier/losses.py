"""loss.py
custom loss functionals
"""
from typing import Optional

import torch

import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Criterion that computes Focal loss. See [1] for details.
    [1] https://arxiv.org/abs/1708.02002

    :param alpha:  weighting factor for the classes (~ to inverse class frequency)
    :type alpha: float
    :param gamma: focusing parameter
    :type gamma: float, optional
    """

    def __init__(self, alpha: float, gamma: Optional[float] = 2.0) -> None:
        """Class constructor method"""
        super(FocalLoss, self).__init__()

        self.alpha: float = alpha
        self.gamma: Optional[float] = gamma
        self.eps: float = 1e-6

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """forward pass

        :params input: input tensor
        :type input: torch.Tensor
        :params target: ground truth/target tensor
        :type target: torch.Tensor
        :return: the value of the FocalLoss evaluated for the given input and
            target tensor
        :rtype: torch.Tensor
        """

        # compute softmax over the classes axis
        input_smax = F.softmax(input, dim=1) + self.eps

        # encode labels as one hot vector
        target_one_hot = F.one_hot(target, num_classes=input.shape[1])

        # compute focal loss
        weight = torch.pow(1.0 - input_smax, self.gamma)
        # might use F.log_softmax here instead
        focal = -self.alpha * weight * torch.log(input_smax)
        loss = torch.sum(target_one_hot * focal, dim=1)

        loss = torch.mean(loss)

        return loss


if __name__ == "__main__":

    pass
