import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s14: torch.SymInt,
        dict_getitem_L_stack0_list_dict_keys_L_stack0_0_: torch.Tensor,
        L_self_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layernorm_eps: torch.Tensor,
        L_self_modules_pooler_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        dict_getitem_l_stack0_list_dict_keys_l_stack0_0_ = (
            dict_getitem_L_stack0_list_dict_keys_L_stack0_0_
        )
        l_self_modules_layernorm_parameters_weight_ = (
            L_self_modules_layernorm_parameters_weight_
        )
        l_self_modules_layernorm_parameters_bias_ = (
            L_self_modules_layernorm_parameters_bias_
        )
        l_self_modules_layernorm_eps = L_self_modules_layernorm_eps
        l_self_modules_pooler_modules_dense_parameters_weight_ = (
            L_self_modules_pooler_modules_dense_parameters_weight_
        )
        l_self_modules_pooler_modules_dense_parameters_bias_ = (
            L_self_modules_pooler_modules_dense_parameters_bias_
        )
        item = l_self_modules_layernorm_eps.item()
        l_self_modules_layernorm_eps = None
        sequence_output = torch.nn.functional.layer_norm(
            dict_getitem_l_stack0_list_dict_keys_l_stack0_0_,
            (192,),
            l_self_modules_layernorm_parameters_weight_,
            l_self_modules_layernorm_parameters_bias_,
            item,
        )
        dict_getitem_l_stack0_list_dict_keys_l_stack0_0_ = (
            l_self_modules_layernorm_parameters_weight_
        ) = l_self_modules_layernorm_parameters_bias_ = item = None
        first_token_tensor = sequence_output[(slice(None, None, None), 0)]
        pooled_output = torch._C._nn.linear(
            first_token_tensor,
            l_self_modules_pooler_modules_dense_parameters_weight_,
            l_self_modules_pooler_modules_dense_parameters_bias_,
        )
        first_token_tensor = (
            l_self_modules_pooler_modules_dense_parameters_weight_
        ) = l_self_modules_pooler_modules_dense_parameters_bias_ = None
        pooled_output_1 = torch.tanh(pooled_output)
        pooled_output = None
        return (sequence_output, pooled_output_1)
