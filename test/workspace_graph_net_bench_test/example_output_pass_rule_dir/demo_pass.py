import torch


def pattern(x, w):
    return torch.nn.functional.relu(torch.nn.functional.linear(x, w))


def replacement_args(x, w):
    return (x, w)


def replacement_func():
    return torch.add
