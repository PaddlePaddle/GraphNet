import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_hidden1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_hidden1_parameters_bias_: torch.nn.parameter.Parameter,
        s12: torch.SymInt,
        L_input_deep_: torch.Tensor,
        L_self_modules_hidden2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_hidden2_parameters_bias_: torch.nn.parameter.Parameter,
        L_input_wide_: torch.Tensor,
        L_self_modules_main_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_main_head_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_aux_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_aux_head_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_hidden1_parameters_weight_ = (
            L_self_modules_hidden1_parameters_weight_
        )
        l_self_modules_hidden1_parameters_bias_ = (
            L_self_modules_hidden1_parameters_bias_
        )
        l_input_deep_ = L_input_deep_
        l_self_modules_hidden2_parameters_weight_ = (
            L_self_modules_hidden2_parameters_weight_
        )
        l_self_modules_hidden2_parameters_bias_ = (
            L_self_modules_hidden2_parameters_bias_
        )
        l_input_wide_ = L_input_wide_
        l_self_modules_main_head_parameters_weight_ = (
            L_self_modules_main_head_parameters_weight_
        )
        l_self_modules_main_head_parameters_bias_ = (
            L_self_modules_main_head_parameters_bias_
        )
        l_self_modules_aux_head_parameters_weight_ = (
            L_self_modules_aux_head_parameters_weight_
        )
        l_self_modules_aux_head_parameters_bias_ = (
            L_self_modules_aux_head_parameters_bias_
        )
        linear = torch._C._nn.linear(
            l_input_deep_,
            l_self_modules_hidden1_parameters_weight_,
            l_self_modules_hidden1_parameters_bias_,
        )
        l_input_deep_ = (
            l_self_modules_hidden1_parameters_weight_
        ) = l_self_modules_hidden1_parameters_bias_ = None
        act = torch.relu(linear)
        linear = None
        linear_1 = torch._C._nn.linear(
            act,
            l_self_modules_hidden2_parameters_weight_,
            l_self_modules_hidden2_parameters_bias_,
        )
        act = (
            l_self_modules_hidden2_parameters_weight_
        ) = l_self_modules_hidden2_parameters_bias_ = None
        act_1 = torch.relu(linear_1)
        linear_1 = None
        concat = torch.cat([l_input_wide_, act_1], dim=1)
        l_input_wide_ = None
        main_output = torch._C._nn.linear(
            concat,
            l_self_modules_main_head_parameters_weight_,
            l_self_modules_main_head_parameters_bias_,
        )
        concat = (
            l_self_modules_main_head_parameters_weight_
        ) = l_self_modules_main_head_parameters_bias_ = None
        aux_output = torch._C._nn.linear(
            act_1,
            l_self_modules_aux_head_parameters_weight_,
            l_self_modules_aux_head_parameters_bias_,
        )
        act_1 = (
            l_self_modules_aux_head_parameters_weight_
        ) = l_self_modules_aux_head_parameters_bias_ = None
        return (main_output, aux_output)
