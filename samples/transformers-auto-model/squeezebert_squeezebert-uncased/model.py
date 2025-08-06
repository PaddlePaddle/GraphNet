import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s63: torch.SymInt,
        dict_getitem_L_stack0_list_dict_keys_L_stack0_0_: torch.Tensor,
        L_self_modules_pooler_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        dict_getitem_l_stack0_list_dict_keys_l_stack0_0_ = (
            dict_getitem_L_stack0_list_dict_keys_L_stack0_0_
        )
        l_self_modules_pooler_modules_dense_parameters_weight_ = (
            L_self_modules_pooler_modules_dense_parameters_weight_
        )
        l_self_modules_pooler_modules_dense_parameters_bias_ = (
            L_self_modules_pooler_modules_dense_parameters_bias_
        )
        first_token_tensor = dict_getitem_l_stack0_list_dict_keys_l_stack0_0_[
            (slice(None, None, None), 0)
        ]
        dict_getitem_l_stack0_list_dict_keys_l_stack0_0_ = None
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
        return (pooled_output_1,)
