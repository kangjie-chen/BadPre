# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: accuracy
@time: 2020/7/11 11:51

"""

from typing import Any, Optional

import torch
from pytorch_lightning.metrics.functional.classification import (
    accuracy
)
from .functional.classification import confusion_matrix
from pytorch_lightning.metrics.metric import TensorMetric


class MaskedAccuracy(TensorMetric):
    """
    Computes the accuracy classification score
    Example:
        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> mask = torch.tensor([1, 1, 1, 0])
        >>> metric = MaskedAccuracy(num_classes=4)
        >>> metric(pred, target, mask)
        tensor(1.)
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        reduction: str = 'elementwise_mean',
        reduce_group: Any = None,
        reduce_op: Any = None,
    ):
        """
        Args:
            num_classes: number of classes
            reduction: a method for reducing accuracies over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='accuracy',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Actual metric computation
        Args:
            pred: predicted labels
            target: ground truth labels
            mask: only calculate metrics where mask==1
        Return:
            A Tensor with the classification score.
        """
        # (yuxian): much faster than pytorch-lightning's implementation
        if self.reduction == "elementwise_mean":
            correct = (pred == target).float() * mask.float()
            return correct.sum() / (mask.float().sum() + 1e-10)

        mask_fill = (1-mask).bool()
        pred = pred.masked_fill(mask=mask_fill, value=-1)
        target = target.masked_fill(mask=mask_fill, value=-1)
        return accuracy(pred=pred, target=target,
                        num_classes=self.num_classes, reduction=self.reduction)


class ConfusionMatrix(TensorMetric):
    """
    Computes the confusion matrix C where each entry C_{i,j} is the number of observations
    in group i that were predicted in group j.

    Example:

        >>> pred = torch.tensor([0, 1, 2, 2])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = ConfusionMatrix()
        >>> metric(pred, target)
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 2.]])

    """

    def __init__(
            self,
            normalize: bool = False,
            reduce_group: Any = None,
            reduce_op: Any = None,
            num_classes: int = None
    ):
        """
        Args:
            normalize: whether to compute a normalized confusion matrix
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='confusion_matrix',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)
        self.normalize = normalize
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Actual metric computation

        Args:
            pred: predicted labels
            target: ground truth labels
            mask: only calculate metrics where mask==1

        Return:
            A Tensor with the confusion matrix.
        """
        pred = pred[mask > 0]
        target = target[mask > 0]
        return confusion_matrix(pred=pred, target=target,
                                normalize=self.normalize, num_classes=self.num_classes)
