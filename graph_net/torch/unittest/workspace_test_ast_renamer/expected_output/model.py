import torch


class GraphModule(torch.nn.Module):
    def forward(self, in_0, w_0):
        tmp_0 = in_0 + w_0
        return tmp_0
