import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_hidden_states_: torch.Tensor,
        L_self_modules_linear_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_linear_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_ln_parameters_weight_ = L_self_modules_ln_parameters_weight_
        l_self_modules_ln_parameters_bias_ = L_self_modules_ln_parameters_bias_
        l_hidden_states_ = L_hidden_states_
        l_self_modules_linear_parameters_weight_ = (
            L_self_modules_linear_parameters_weight_
        )
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_
        hidden_states = torch.nn.functional.layer_norm(
            l_hidden_states_,
            (2560,),
            l_self_modules_ln_parameters_weight_,
            l_self_modules_ln_parameters_bias_,
            1e-05,
        )
        l_hidden_states_ = (
            l_self_modules_ln_parameters_weight_
        ) = l_self_modules_ln_parameters_bias_ = None
        linear = torch._C._nn.linear(
            hidden_states,
            l_self_modules_linear_parameters_weight_,
            l_self_modules_linear_parameters_bias_,
        )
        hidden_states = (
            l_self_modules_linear_parameters_weight_
        ) = l_self_modules_linear_parameters_bias_ = None
        logits = linear.to(torch.float32)
        linear = None
        return (logits,)
