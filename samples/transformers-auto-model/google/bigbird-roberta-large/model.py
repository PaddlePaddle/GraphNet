import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        dict_getitem_L_stack0_list_dict_keys_L_stack0_0_: torch.Tensor,
        L_self_modules_qa_classifier_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_qa_classifier_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_qa_classifier_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_qa_classifier_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_qa_classifier_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_qa_classifier_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_qa_classifier_modules_qa_outputs_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_qa_classifier_modules_qa_outputs_parameters_bias_: torch.nn.parameter.Parameter,
        L_logits_mask_: torch.Tensor,
    ):
        dict_getitem_l_stack0_list_dict_keys_l_stack0_0_ = (
            dict_getitem_L_stack0_list_dict_keys_L_stack0_0_
        )
        l_self_modules_qa_classifier_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_qa_classifier_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_qa_classifier_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_qa_classifier_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_qa_classifier_modules_output_modules_dense_parameters_weight_ = (
            L_self_modules_qa_classifier_modules_output_modules_dense_parameters_weight_
        )
        l_self_modules_qa_classifier_modules_output_modules_dense_parameters_bias_ = (
            L_self_modules_qa_classifier_modules_output_modules_dense_parameters_bias_
        )
        l_self_modules_qa_classifier_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_qa_classifier_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_qa_classifier_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_qa_classifier_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_qa_classifier_modules_qa_outputs_parameters_weight_ = (
            L_self_modules_qa_classifier_modules_qa_outputs_parameters_weight_
        )
        l_self_modules_qa_classifier_modules_qa_outputs_parameters_bias_ = (
            L_self_modules_qa_classifier_modules_qa_outputs_parameters_bias_
        )
        l_logits_mask_ = L_logits_mask_
        hidden_states = torch.nn.functional.dropout(
            dict_getitem_l_stack0_list_dict_keys_l_stack0_0_, 0.1, False, False
        )
        hidden_states_1 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_qa_classifier_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_qa_classifier_modules_intermediate_modules_dense_parameters_bias_,
        )
        hidden_states = l_self_modules_qa_classifier_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_qa_classifier_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul = 0.5 * hidden_states_1
        pow_1 = torch.pow(hidden_states_1, 3.0)
        mul_1 = 0.044715 * pow_1
        pow_1 = None
        add = hidden_states_1 + mul_1
        hidden_states_1 = mul_1 = None
        mul_2 = 0.7978845608028654 * add
        add = None
        tanh = torch.tanh(mul_2)
        mul_2 = None
        add_1 = 1.0 + tanh
        tanh = None
        hidden_states_2 = mul * add_1
        mul = add_1 = None
        hidden_states_3 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_qa_classifier_modules_output_modules_dense_parameters_weight_,
            l_self_modules_qa_classifier_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_2 = (
            l_self_modules_qa_classifier_modules_output_modules_dense_parameters_weight_
        ) = (
            l_self_modules_qa_classifier_modules_output_modules_dense_parameters_bias_
        ) = None
        hidden_states_4 = torch.nn.functional.dropout(
            hidden_states_3, 0.1, False, False
        )
        hidden_states_3 = None
        add_2 = hidden_states_4 + dict_getitem_l_stack0_list_dict_keys_l_stack0_0_
        hidden_states_4 = dict_getitem_l_stack0_list_dict_keys_l_stack0_0_ = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            add_2,
            (1024,),
            l_self_modules_qa_classifier_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_qa_classifier_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_2 = l_self_modules_qa_classifier_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_qa_classifier_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_6 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_qa_classifier_modules_qa_outputs_parameters_weight_,
            l_self_modules_qa_classifier_modules_qa_outputs_parameters_bias_,
        )
        hidden_states_5 = (
            l_self_modules_qa_classifier_modules_qa_outputs_parameters_weight_
        ) = l_self_modules_qa_classifier_modules_qa_outputs_parameters_bias_ = None
        mul_4 = l_logits_mask_ * 1000000.0
        l_logits_mask_ = None
        logits = hidden_states_6 - mul_4
        hidden_states_6 = mul_4 = None
        split = logits.split(1, dim=-1)
        logits = None
        start_logits = split[0]
        end_logits = split[1]
        split = None
        squeeze = start_logits.squeeze(-1)
        start_logits = None
        start_logits_1 = squeeze.contiguous()
        squeeze = None
        squeeze_1 = end_logits.squeeze(-1)
        end_logits = None
        end_logits_1 = squeeze_1.contiguous()
        squeeze_1 = None
        return (start_logits_1, end_logits_1)
