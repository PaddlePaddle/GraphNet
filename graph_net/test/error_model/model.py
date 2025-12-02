import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        add_22,
        extended_attention_mask_2,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_,
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_,
    ):
        hidden_states_66 = torch.nn.functional.layer_norm(
            add_22,
            (32,),
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_22 = l_l_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = l_l_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_44 = torch.nn.functional.linear(
            hidden_states_66,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_16 = linear_44.view(2, -1, 4, 8)
        linear_44 = None
        query_layer_4 = view_16.transpose(1, 2)
        view_16 = None
        linear_45 = torch.nn.functional.linear(
            hidden_states_66,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_17 = linear_45.view(2, -1, 4, 8)
        linear_45 = None
        key_layer_4 = view_17.transpose(1, 2)
        view_17 = None
        linear_46 = torch.nn.functional.linear(
            hidden_states_66,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_18 = linear_46.view(2, -1, 4, 8)
        linear_46 = None
        value_layer_4 = view_18.transpose(1, 2)
        view_18 = None
        transpose_25 = key_layer_4.transpose(-1, -2)
        key_layer_4 = None
        attention_scores_22 = torch.matmul(query_layer_4, transpose_25)
        query_layer_4 = transpose_25 = None
        attention_scores_23 = attention_scores_22 / 2.8284271247461903
        attention_scores_22 = None
        eps = torch.tensor(1e-8, device=attention_scores_23.device)
        nan_val = eps / (eps - eps)
        attention_scores_23 = attention_scores_23 + nan_val
        nan_val = None
        to_8 = extended_attention_mask_2.to(device(type="cuda", index=0))
        extended_attention_mask_2 = None
        attention_scores_24 = attention_scores_23 + to_8
        attention_scores_23 = to_8 = None
        _log_api_usage_once_4 = torch._C._log_api_usage_once("python.nn_module")
        _log_api_usage_once_4 = None
        attention_probs_14 = torch.nn.functional.softmax(
            attention_scores_24, -1, _stacklevel=5
        )
        attention_scores_24 = None
        attention_probs_dropped_4 = torch.nn.functional.dropout(
            attention_probs_14, 0.0, False, False
        )
        attention_probs_14 = None
        context_layer_22 = torch.matmul(attention_probs_dropped_4, value_layer_4)
        attention_probs_dropped_4 = value_layer_4 = None
        permute_14 = context_layer_22.permute(0, 2, 1, 3)
        context_layer_22 = None
        context_layer_23 = permute_14.contiguous()
        permute_14 = None
        context_layer_24 = context_layer_23.view(2, 14, 32)
        context_layer_23 = None
        hidden_states_67 = torch.nn.functional.linear(
            context_layer_24,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_24 = l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_68 = torch.nn.functional.dropout(
            hidden_states_67, 0.0, False, False
        )
        hidden_states_67 = None
        add_24 = hidden_states_68 + hidden_states_66
        hidden_states_68 = hidden_states_66 = None
        hidden_states_69 = torch.nn.functional.layer_norm(
            add_24,
            (32,),
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_24 = l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_70 = torch.nn.functional.linear(
            hidden_states_69,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_71 = torch.nn.functional.gelu(hidden_states_70)
        hidden_states_70 = None
        hidden_states_72 = torch.nn.functional.linear(
            hidden_states_71,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_71 = l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_73 = torch.nn.functional.dropout(
            hidden_states_72, 0.0, False, False
        )
        hidden_states_72 = None
        nan_val = torch.tensor(0.0, device=hidden_states_73.device) / torch.tensor(
            0.0, device=hidden_states_73.device
        )
        hidden_states_73 = hidden_states_73 + nan_val
        nan_val = None
        add_25 = hidden_states_73 + hidden_states_69
        hidden_states_73 = hidden_states_69 = None
        hidden_states_74 = torch.nn.functional.layer_norm(
            add_25,
            (32,),
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_,
            l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_25 = l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = l_l_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = (None)
        return (hidden_states_74,)
