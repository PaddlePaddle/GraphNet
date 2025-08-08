import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        s77: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc4_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_fc1_parameters_weight_ = L_self_modules_fc1_parameters_weight_
        l_self_modules_fc1_parameters_bias_ = L_self_modules_fc1_parameters_bias_
        l_x_ = L_x_
        l_self_modules_fc2_parameters_weight_ = L_self_modules_fc2_parameters_weight_
        l_self_modules_fc2_parameters_bias_ = L_self_modules_fc2_parameters_bias_
        l_self_modules_fc3_parameters_weight_ = L_self_modules_fc3_parameters_weight_
        l_self_modules_fc3_parameters_bias_ = L_self_modules_fc3_parameters_bias_
        l_self_modules_fc4_parameters_weight_ = L_self_modules_fc4_parameters_weight_
        l_self_modules_fc4_parameters_bias_ = L_self_modules_fc4_parameters_bias_
        linear = torch._C._nn.linear(
            l_x_,
            l_self_modules_fc1_parameters_weight_,
            l_self_modules_fc1_parameters_bias_,
        )
        l_x_ = (
            l_self_modules_fc1_parameters_weight_
        ) = l_self_modules_fc1_parameters_bias_ = None
        act = torch.relu(linear)
        linear = None
        linear_1 = torch._C._nn.linear(
            act,
            l_self_modules_fc2_parameters_weight_,
            l_self_modules_fc2_parameters_bias_,
        )
        act = (
            l_self_modules_fc2_parameters_weight_
        ) = l_self_modules_fc2_parameters_bias_ = None
        act_1 = torch.relu(linear_1)
        linear_1 = None
        linear_2 = torch._C._nn.linear(
            act_1,
            l_self_modules_fc3_parameters_weight_,
            l_self_modules_fc3_parameters_bias_,
        )
        act_1 = (
            l_self_modules_fc3_parameters_weight_
        ) = l_self_modules_fc3_parameters_bias_ = None
        act_2 = torch.relu(linear_2)
        linear_2 = None
        linear_3 = torch._C._nn.linear(
            act_2,
            l_self_modules_fc4_parameters_weight_,
            l_self_modules_fc4_parameters_bias_,
        )
        act_2 = (
            l_self_modules_fc4_parameters_weight_
        ) = l_self_modules_fc4_parameters_bias_ = None
        return (linear_3,)
