import torch


class GraphModule(torch.nn.Module):
    def forward(self, x_data, L__self___weight):
        res = x_data + L__self___weight
        L__self___weight = None
        return res
