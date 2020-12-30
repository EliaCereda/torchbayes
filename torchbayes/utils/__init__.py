# TODO: move to proper utils files
import numpy as np
import torch


def take(it, n):
    for x, _ in zip(it, range(n)):
        yield x


def heterogeneous_transpose(x, stack=None, dim=0):
    """
    Transpose a list of lists of tensors of heterogeneous shapes. A possible use
    of this function is to record a certain number of metrics over time and then
    extract a single tensor per metric.

    In this case, the shape of the input should be interpreted as
        time steps ✕ metrics ✕ (shape of each metric)
    Every time step must record the same number of metrics and each metric can
    have its own shape. Every metric must have the same shape (or broadcastable
    shape) over all time steps.

    The resulting output is a list of tensors with shape
        metrics ✕ (time steps ✕ shape of each metric)

    :param x: list of tensors to be transposed.
    :param stack: whether to use `torch.stack` or `torch.cat` to construct the
    final tensors. When `None`, the default, `torch.stack` is used for zero-rank
    tensors and `torch.cat` otherwise.
    :param dim: dimension to be stacked or concatenated.
    :return: the transposed tensors.
    """
    transposed = np.asarray(x, dtype=object).T
    slices = []

    for slice in transposed:
        # Assumes that if the first element is zero-rank, then all are.
        if stack or (stack is None and len(slice) > 0 and slice[0].ndim == 0):
            slice = torch.stack(list(slice), dim)
        else:
            slice = torch.cat(list(slice), dim)

        slices.append(slice)

    return slices


### Helper functions from PyTorch

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x

        return tuple(repeat(x, n))


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
