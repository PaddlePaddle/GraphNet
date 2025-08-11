import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s40: torch.SymInt,
        L_stack0_0_: torch.Tensor,
        L_self_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_output_modules_LayerNorm_variance_epsilon: torch.Tensor,
        L_self_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_stack0_0_ = L_stack0_0_
        l_self_modules_intermediate_modules_dense_parameters_weight_ = (
            L_self_modules_intermediate_modules_dense_parameters_weight_
        )
        l_self_modules_intermediate_modules_dense_parameters_bias_ = (
            L_self_modules_intermediate_modules_dense_parameters_bias_
        )
        l_self_modules_output_modules_dense_parameters_weight_ = (
            L_self_modules_output_modules_dense_parameters_weight_
        )
        l_self_modules_output_modules_dense_parameters_bias_ = (
            L_self_modules_output_modules_dense_parameters_bias_
        )
        l_self_modules_output_modules_dropout_p = (
            L_self_modules_output_modules_dropout_p
        )
        l_self_modules_output_modules_layer_norm_variance_epsilon = (
            L_self_modules_output_modules_LayerNorm_variance_epsilon
        )
        l_self_modules_output_modules_layer_norm_parameters_weight_ = (
            L_self_modules_output_modules_LayerNorm_parameters_weight_
        )
        l_self_modules_output_modules_layer_norm_parameters_bias_ = (
            L_self_modules_output_modules_LayerNorm_parameters_bias_
        )
        hidden_states = torch._C._nn.linear(
            l_stack0_0_,
            l_self_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_intermediate_modules_dense_parameters_weight_ = (
            l_self_modules_intermediate_modules_dense_parameters_bias_
        ) = None
        hidden_states_1 = torch._C._nn.gelu(hidden_states)
        hidden_states = None
        hidden_states_2 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_output_modules_dense_parameters_weight_,
            l_self_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_1 = (
            l_self_modules_output_modules_dense_parameters_weight_
        ) = l_self_modules_output_modules_dense_parameters_bias_ = None
        item = l_self_modules_output_modules_dropout_p.item()
        l_self_modules_output_modules_dropout_p = None
        hidden_states_3 = torch.nn.functional.dropout(
            hidden_states_2, item, False, False
        )
        hidden_states_2 = item = None
        add = hidden_states_3 + l_stack0_0_
        hidden_states_3 = l_stack0_0_ = None
        hidden_states_4 = add.float()
        add = None
        mean = hidden_states_4.mean(-1, keepdim=True)
        sub = hidden_states_4 - mean
        pow_1 = sub.pow(2)
        sub = None
        variance = pow_1.mean(-1, keepdim=True)
        pow_1 = None
        sub_1 = hidden_states_4 - mean
        hidden_states_4 = mean = None
        item_1 = l_self_modules_output_modules_layer_norm_variance_epsilon.item()
        l_self_modules_output_modules_layer_norm_variance_epsilon = None
        add_1 = variance + item_1
        variance = item_1 = None
        sqrt = torch.sqrt(add_1)
        add_1 = None
        hidden_states_5 = sub_1 / sqrt
        sub_1 = sqrt = None
        hidden_states_6 = hidden_states_5.to(torch.float32)
        hidden_states_5 = None
        mul = (
            l_self_modules_output_modules_layer_norm_parameters_weight_
            * hidden_states_6
        )
        l_self_modules_output_modules_layer_norm_parameters_weight_ = (
            hidden_states_6
        ) = None
        y = mul + l_self_modules_output_modules_layer_norm_parameters_bias_
        mul = l_self_modules_output_modules_layer_norm_parameters_bias_ = None
        return (y,)
