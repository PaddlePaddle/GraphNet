import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        dict_getitem_L_stack0_list_dict_keys_L_stack0_0_: torch.Tensor,
        L_self_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
    ):
        dict_getitem_l_stack0_list_dict_keys_l_stack0_0_ = (
            dict_getitem_L_stack0_list_dict_keys_L_stack0_0_
        )
        l_self_modules_layer_norm_parameters_weight_ = (
            L_self_modules_LayerNorm_parameters_weight_
        )
        l_self_modules_layer_norm_parameters_bias_ = (
            L_self_modules_LayerNorm_parameters_bias_
        )
        l_attention_mask_ = L_attention_mask_
        sequence_output = torch.nn.functional.layer_norm(
            dict_getitem_l_stack0_list_dict_keys_l_stack0_0_,
            (768,),
            l_self_modules_layer_norm_parameters_weight_,
            l_self_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        dict_getitem_l_stack0_list_dict_keys_l_stack0_0_ = (
            l_self_modules_layer_norm_parameters_weight_
        ) = l_self_modules_layer_norm_parameters_bias_ = None
        unsqueeze = l_attention_mask_.unsqueeze(-1)
        l_attention_mask_ = None
        expand_as = unsqueeze.expand_as(sequence_output)
        unsqueeze = None
        attention_mask = expand_as.float()
        expand_as = None
        mul = sequence_output * attention_mask
        sum_embeddings = torch.sum(mul, dim=1)
        mul = None
        sum_2 = attention_mask.sum(dim=1)
        attention_mask = None
        sum_mask = torch.clamp(sum_2, min=1e-09)
        sum_2 = None
        embedding = sum_embeddings / sum_mask
        sum_embeddings = sum_mask = None
        return (sequence_output, embedding)
