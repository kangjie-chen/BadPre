# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: classification
@time: 2020/7/11 17:44

"""

from pytorch_lightning.metrics.functional.classification import *


def confusion_matrix(
        pred: torch.Tensor,
        target: torch.Tensor,
        normalize: bool = False,
        num_classes: int = None
) -> torch.Tensor:
    """
    Computes the confusion matrix C where each entry C_{i,j} is the number of observations
    in group i that were predicted in group j.

    Args:
        pred: estimated targets
        target: ground truth labels
        normalize: normalizes confusion matrix
        num_classes: int, num_labels

    Return:
        Tensor, confusion matrix C [num_classes, num_classes ]

    Example:

        >>> x = torch.tensor([1, 2, 3])
        >>> y = torch.tensor([0, 2, 3])
        >>> confusion_matrix(x, y)
        tensor([[0., 1., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]])
    """
    num_classes = get_num_classes(pred, target, num_classes=num_classes)

    unique_labels = target.view(-1) * num_classes + pred.view(-1)

    bins = torch.bincount(unique_labels, minlength=num_classes ** 2)
    cm = bins.reshape(num_classes, num_classes).squeeze().float()

    if normalize:
        cm = cm / cm.sum(-1)

    return cm
