import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_parameters_weight_: torch.nn.parameter.Parameter,
        L_x_: torch.Tensor,
    ):
        l_self_parameters_weight_ = L_self_parameters_weight_
        l_x_ = L_x_
        linear = torch._C._nn.linear(l_x_, l_self_parameters_weight_, None)
        l_x_ = l_self_parameters_weight_ = None
        return (linear,)
