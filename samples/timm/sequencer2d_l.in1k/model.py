import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s0: torch.SymInt,
        L_stack0_: torch.Tensor,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_stack0_ = L_stack0_
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = l_stack0_.mean((1, 2), keepdim=False)
        l_stack0_ = None
        x_1 = torch.nn.functional.dropout(x, 0.0, False, False)
        x = None
        x_2 = torch._C._nn.linear(
            x_1,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_1 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_2,)
