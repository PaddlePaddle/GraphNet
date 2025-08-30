import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_modules_rotary_emb_buffers_inv_freq_: torch.Tensor,
        L_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_embeds_ = L_inputs_embeds_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_rotary_emb_buffers_inv_freq_ = (
            L_self_modules_rotary_emb_buffers_inv_freq_
        )
        l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_0_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_1_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_2_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_3_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_4_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_5_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_6_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_6_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_6_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_6_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_6_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_7_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_7_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_7_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_7_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_7_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_8_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_8_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_8_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_8_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_8_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_9_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_9_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_9_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_9_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_9_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_10_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_10_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_10_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_10_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_10_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_11_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_11_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_11_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_11_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_11_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_12_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_12_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_12_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_12_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_12_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_13_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_13_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_13_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_13_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_13_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_14_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_14_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_14_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_14_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_14_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_layers_modules_15_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_layers_modules_15_modules_attention_modules_dense_parameters_weight_ = L_self_modules_layers_modules_15_modules_attention_modules_dense_parameters_weight_
        l_self_modules_layers_modules_15_modules_attention_modules_dense_parameters_bias_ = L_self_modules_layers_modules_15_modules_attention_modules_dense_parameters_bias_
        l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_final_layer_norm_parameters_bias_
        )
        cache_position = torch.arange(0, 2, device=device(type="cuda", index=0))
        position_ids = cache_position.unsqueeze(0)
        attention_mask = l_attention_mask_.to(
            device=device(type="cuda", index=0), dtype=torch.bool
        )
        l_attention_mask_ = None
        mask_indices = torch.arange(2, device=device(type="cuda", index=0))
        mask_indices += 0
        mask_indices_1 = mask_indices
        mask_indices = None
        local_padding_mask = attention_mask[(slice(None, None, None), mask_indices_1)]
        attention_mask = mask_indices_1 = None
        kv_arange = torch.arange(2, device=device(type="cuda", index=0))
        kv_arange += 0
        kv_arange_1 = kv_arange
        kv_arange = None
        reshaped_cache_position = cache_position.view(-1, 1)
        cache_position = None
        causal_mask = kv_arange_1 <= reshaped_cache_position
        kv_arange_1 = reshaped_cache_position = None
        getitem_1 = causal_mask[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        causal_mask = None
        causal_mask_1 = getitem_1.expand(1, -1, -1, -1)
        getitem_1 = None
        getitem_2 = local_padding_mask[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        local_padding_mask = None
        causal_mask_2 = causal_mask_1 * getitem_2
        causal_mask_1 = getitem_2 = None
        hidden_states = torch.nn.functional.dropout(l_inputs_embeds_, 0.0, False, False)
        l_inputs_embeds_ = None
        _set_grad_enabled = torch._C._set_grad_enabled(False)
        _set_grad_enabled = None
        getitem_3 = l_self_modules_rotary_emb_buffers_inv_freq_[
            (None, slice(None, None, None), None)
        ]
        l_self_modules_rotary_emb_buffers_inv_freq_ = None
        float_1 = getitem_3.float()
        getitem_3 = None
        expand_1 = float_1.expand(1, -1, 1)
        float_1 = None
        inv_freq_expanded = expand_1.to(device(type="cuda", index=0))
        expand_1 = None
        getitem_4 = position_ids[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        position_ids = None
        position_ids_expanded = getitem_4.float()
        getitem_4 = None
        float_3 = inv_freq_expanded.float()
        inv_freq_expanded = None
        float_4 = position_ids_expanded.float()
        position_ids_expanded = None
        matmul = float_3 @ float_4
        float_3 = float_4 = None
        freqs = matmul.transpose(1, 2)
        matmul = None
        emb = torch.cat((freqs, freqs), dim=-1)
        freqs = None
        cos = emb.cos()
        cos_1 = cos * 1.0
        cos = None
        sin = emb.sin()
        emb = None
        sin_1 = sin * 1.0
        sin = None
        cos_2 = cos_1.to(dtype=torch.float16)
        cos_1 = None
        sin_2 = sin_1.to(dtype=torch.float16)
        sin_1 = None
        _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
        _set_grad_enabled_1 = None
        layer_norm = torch.nn.functional.layer_norm(
            hidden_states,
            (2048,),
            l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_0_modules_input_layernorm_parameters_bias_
        ) = None
        linear = torch._C._nn.linear(
            layer_norm,
            l_self_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm = l_self_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_1 = linear.view((1, 2, -1, 768))
        linear = None
        qkv = view_1.transpose(1, 2)
        view_1 = None
        chunk = qkv.chunk(3, dim=-1)
        qkv = None
        query_states = chunk[0]
        key_states = chunk[1]
        value_states = chunk[2]
        chunk = None
        cos_3 = cos_2.unsqueeze(1)
        sin_3 = sin_2.unsqueeze(1)
        q_rot = query_states[(Ellipsis, slice(None, 64, None))]
        q_pass = query_states[(Ellipsis, slice(64, None, None))]
        query_states = None
        k_rot = key_states[(Ellipsis, slice(None, 64, None))]
        k_pass = key_states[(Ellipsis, slice(64, None, None))]
        key_states = None
        mul_3 = q_rot * cos_3
        x1 = q_rot[(Ellipsis, slice(None, 32, None))]
        x2 = q_rot[(Ellipsis, slice(32, None, None))]
        q_rot = None
        neg = -x2
        x2 = None
        cat_1 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_4 = cat_1 * sin_3
        cat_1 = None
        q_embed = mul_3 + mul_4
        mul_3 = mul_4 = None
        mul_5 = k_rot * cos_3
        cos_3 = None
        x1_1 = k_rot[(Ellipsis, slice(None, 32, None))]
        x2_1 = k_rot[(Ellipsis, slice(32, None, None))]
        k_rot = None
        neg_1 = -x2_1
        x2_1 = None
        cat_2 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_6 = cat_2 * sin_3
        cat_2 = sin_3 = None
        k_embed = mul_5 + mul_6
        mul_5 = mul_6 = None
        q_embed_1 = torch.cat([q_embed, q_pass], dim=-1)
        q_embed = q_pass = None
        k_embed_1 = torch.cat([k_embed, k_pass], dim=-1)
        k_embed = k_pass = None
        attention_mask_1 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query = q_embed_1.contiguous()
        q_embed_1 = None
        key = k_embed_1.contiguous()
        value = value_states.contiguous()
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query = key = value = attention_mask_1 = None
        transpose_2 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_2.contiguous()
        transpose_2 = None
        reshape = attn_output_1.reshape(1, 2, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_2 = l_self_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_4 = torch.nn.functional.dropout(attn_output_3, 0.0, False, False)
        attn_output_3 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            hidden_states,
            (2048,),
            l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_1 = torch._C._nn.linear(
            layer_norm_1,
            l_self_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_1 = l_self_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_2 = torch._C._nn.gelu(hidden_states_1)
        hidden_states_1 = None
        hidden_states_3 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output = torch.nn.functional.dropout(hidden_states_3, 0.0, False, False)
        hidden_states_3 = None
        add_2 = mlp_output + attn_output_4
        mlp_output = attn_output_4 = None
        hidden_states_4 = add_2 + hidden_states
        add_2 = hidden_states = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (2048,),
            l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_1_modules_input_layernorm_parameters_bias_
        ) = None
        linear_4 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_2 = l_self_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_2 = linear_4.view((1, 2, -1, 768))
        linear_4 = None
        qkv_1 = view_2.transpose(1, 2)
        view_2 = None
        chunk_1 = qkv_1.chunk(3, dim=-1)
        qkv_1 = None
        query_states_1 = chunk_1[0]
        key_states_1 = chunk_1[1]
        value_states_1 = chunk_1[2]
        chunk_1 = None
        cos_4 = cos_2.unsqueeze(1)
        sin_4 = sin_2.unsqueeze(1)
        q_rot_1 = query_states_1[(Ellipsis, slice(None, 64, None))]
        q_pass_1 = query_states_1[(Ellipsis, slice(64, None, None))]
        query_states_1 = None
        k_rot_1 = key_states_1[(Ellipsis, slice(None, 64, None))]
        k_pass_1 = key_states_1[(Ellipsis, slice(64, None, None))]
        key_states_1 = None
        mul_7 = q_rot_1 * cos_4
        x1_2 = q_rot_1[(Ellipsis, slice(None, 32, None))]
        x2_2 = q_rot_1[(Ellipsis, slice(32, None, None))]
        q_rot_1 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_5 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_8 = cat_5 * sin_4
        cat_5 = None
        q_embed_2 = mul_7 + mul_8
        mul_7 = mul_8 = None
        mul_9 = k_rot_1 * cos_4
        cos_4 = None
        x1_3 = k_rot_1[(Ellipsis, slice(None, 32, None))]
        x2_3 = k_rot_1[(Ellipsis, slice(32, None, None))]
        k_rot_1 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_6 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_10 = cat_6 * sin_4
        cat_6 = sin_4 = None
        k_embed_2 = mul_9 + mul_10
        mul_9 = mul_10 = None
        q_embed_3 = torch.cat([q_embed_2, q_pass_1], dim=-1)
        q_embed_2 = q_pass_1 = None
        k_embed_3 = torch.cat([k_embed_2, k_pass_1], dim=-1)
        k_embed_2 = k_pass_1 = None
        attention_mask_2 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_1 = q_embed_3.contiguous()
        q_embed_3 = None
        key_1 = k_embed_3.contiguous()
        value_1 = value_states_1.contiguous()
        attn_output_5 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = attention_mask_2 = None
        transpose_4 = attn_output_5.transpose(1, 2)
        attn_output_5 = None
        attn_output_6 = transpose_4.contiguous()
        transpose_4 = None
        reshape_1 = attn_output_6.reshape(1, 2, -1)
        attn_output_6 = None
        attn_output_7 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_8 = torch._C._nn.linear(
            attn_output_7,
            l_self_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_7 = l_self_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_9 = torch.nn.functional.dropout(attn_output_8, 0.0, False, False)
        attn_output_8 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (2048,),
            l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_5 = torch._C._nn.linear(
            layer_norm_3,
            l_self_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_3 = l_self_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_6 = torch._C._nn.gelu(hidden_states_5)
        hidden_states_5 = None
        hidden_states_7 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_6 = l_self_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_1 = torch.nn.functional.dropout(hidden_states_7, 0.0, False, False)
        hidden_states_7 = None
        add_6 = mlp_output_1 + attn_output_9
        mlp_output_1 = attn_output_9 = None
        hidden_states_8 = add_6 + hidden_states_4
        add_6 = hidden_states_4 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            hidden_states_8,
            (2048,),
            l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_2_modules_input_layernorm_parameters_bias_
        ) = None
        linear_8 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_4 = l_self_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_3 = linear_8.view((1, 2, -1, 768))
        linear_8 = None
        qkv_2 = view_3.transpose(1, 2)
        view_3 = None
        chunk_2 = qkv_2.chunk(3, dim=-1)
        qkv_2 = None
        query_states_2 = chunk_2[0]
        key_states_2 = chunk_2[1]
        value_states_2 = chunk_2[2]
        chunk_2 = None
        cos_5 = cos_2.unsqueeze(1)
        sin_5 = sin_2.unsqueeze(1)
        q_rot_2 = query_states_2[(Ellipsis, slice(None, 64, None))]
        q_pass_2 = query_states_2[(Ellipsis, slice(64, None, None))]
        query_states_2 = None
        k_rot_2 = key_states_2[(Ellipsis, slice(None, 64, None))]
        k_pass_2 = key_states_2[(Ellipsis, slice(64, None, None))]
        key_states_2 = None
        mul_11 = q_rot_2 * cos_5
        x1_4 = q_rot_2[(Ellipsis, slice(None, 32, None))]
        x2_4 = q_rot_2[(Ellipsis, slice(32, None, None))]
        q_rot_2 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_9 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_12 = cat_9 * sin_5
        cat_9 = None
        q_embed_4 = mul_11 + mul_12
        mul_11 = mul_12 = None
        mul_13 = k_rot_2 * cos_5
        cos_5 = None
        x1_5 = k_rot_2[(Ellipsis, slice(None, 32, None))]
        x2_5 = k_rot_2[(Ellipsis, slice(32, None, None))]
        k_rot_2 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_10 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_14 = cat_10 * sin_5
        cat_10 = sin_5 = None
        k_embed_4 = mul_13 + mul_14
        mul_13 = mul_14 = None
        q_embed_5 = torch.cat([q_embed_4, q_pass_2], dim=-1)
        q_embed_4 = q_pass_2 = None
        k_embed_5 = torch.cat([k_embed_4, k_pass_2], dim=-1)
        k_embed_4 = k_pass_2 = None
        attention_mask_3 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_2 = q_embed_5.contiguous()
        q_embed_5 = None
        key_2 = k_embed_5.contiguous()
        value_2 = value_states_2.contiguous()
        attn_output_10 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = attention_mask_3 = None
        transpose_6 = attn_output_10.transpose(1, 2)
        attn_output_10 = None
        attn_output_11 = transpose_6.contiguous()
        transpose_6 = None
        reshape_2 = attn_output_11.reshape(1, 2, -1)
        attn_output_11 = None
        attn_output_12 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_13 = torch._C._nn.linear(
            attn_output_12,
            l_self_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_12 = l_self_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_14 = torch.nn.functional.dropout(attn_output_13, 0.0, False, False)
        attn_output_13 = None
        layer_norm_5 = torch.nn.functional.layer_norm(
            hidden_states_8,
            (2048,),
            l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_9 = torch._C._nn.linear(
            layer_norm_5,
            l_self_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_5 = l_self_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_10 = torch._C._nn.gelu(hidden_states_9)
        hidden_states_9 = None
        hidden_states_11 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_10 = l_self_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_2 = torch.nn.functional.dropout(hidden_states_11, 0.0, False, False)
        hidden_states_11 = None
        add_10 = mlp_output_2 + attn_output_14
        mlp_output_2 = attn_output_14 = None
        hidden_states_12 = add_10 + hidden_states_8
        add_10 = hidden_states_8 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            hidden_states_12,
            (2048,),
            l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_3_modules_input_layernorm_parameters_bias_
        ) = None
        linear_12 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_6 = l_self_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_4 = linear_12.view((1, 2, -1, 768))
        linear_12 = None
        qkv_3 = view_4.transpose(1, 2)
        view_4 = None
        chunk_3 = qkv_3.chunk(3, dim=-1)
        qkv_3 = None
        query_states_3 = chunk_3[0]
        key_states_3 = chunk_3[1]
        value_states_3 = chunk_3[2]
        chunk_3 = None
        cos_6 = cos_2.unsqueeze(1)
        sin_6 = sin_2.unsqueeze(1)
        q_rot_3 = query_states_3[(Ellipsis, slice(None, 64, None))]
        q_pass_3 = query_states_3[(Ellipsis, slice(64, None, None))]
        query_states_3 = None
        k_rot_3 = key_states_3[(Ellipsis, slice(None, 64, None))]
        k_pass_3 = key_states_3[(Ellipsis, slice(64, None, None))]
        key_states_3 = None
        mul_15 = q_rot_3 * cos_6
        x1_6 = q_rot_3[(Ellipsis, slice(None, 32, None))]
        x2_6 = q_rot_3[(Ellipsis, slice(32, None, None))]
        q_rot_3 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_13 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_16 = cat_13 * sin_6
        cat_13 = None
        q_embed_6 = mul_15 + mul_16
        mul_15 = mul_16 = None
        mul_17 = k_rot_3 * cos_6
        cos_6 = None
        x1_7 = k_rot_3[(Ellipsis, slice(None, 32, None))]
        x2_7 = k_rot_3[(Ellipsis, slice(32, None, None))]
        k_rot_3 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_14 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_18 = cat_14 * sin_6
        cat_14 = sin_6 = None
        k_embed_6 = mul_17 + mul_18
        mul_17 = mul_18 = None
        q_embed_7 = torch.cat([q_embed_6, q_pass_3], dim=-1)
        q_embed_6 = q_pass_3 = None
        k_embed_7 = torch.cat([k_embed_6, k_pass_3], dim=-1)
        k_embed_6 = k_pass_3 = None
        attention_mask_4 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_3 = q_embed_7.contiguous()
        q_embed_7 = None
        key_3 = k_embed_7.contiguous()
        value_3 = value_states_3.contiguous()
        attn_output_15 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = attention_mask_4 = None
        transpose_8 = attn_output_15.transpose(1, 2)
        attn_output_15 = None
        attn_output_16 = transpose_8.contiguous()
        transpose_8 = None
        reshape_3 = attn_output_16.reshape(1, 2, -1)
        attn_output_16 = None
        attn_output_17 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_18 = torch._C._nn.linear(
            attn_output_17,
            l_self_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_17 = l_self_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_19 = torch.nn.functional.dropout(attn_output_18, 0.0, False, False)
        attn_output_18 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            hidden_states_12,
            (2048,),
            l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_13 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_7 = l_self_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_14 = torch._C._nn.gelu(hidden_states_13)
        hidden_states_13 = None
        hidden_states_15 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_14 = l_self_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_3 = torch.nn.functional.dropout(hidden_states_15, 0.0, False, False)
        hidden_states_15 = None
        add_14 = mlp_output_3 + attn_output_19
        mlp_output_3 = attn_output_19 = None
        hidden_states_16 = add_14 + hidden_states_12
        add_14 = hidden_states_12 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            hidden_states_16,
            (2048,),
            l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_4_modules_input_layernorm_parameters_bias_
        ) = None
        linear_16 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_8 = l_self_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_5 = linear_16.view((1, 2, -1, 768))
        linear_16 = None
        qkv_4 = view_5.transpose(1, 2)
        view_5 = None
        chunk_4 = qkv_4.chunk(3, dim=-1)
        qkv_4 = None
        query_states_4 = chunk_4[0]
        key_states_4 = chunk_4[1]
        value_states_4 = chunk_4[2]
        chunk_4 = None
        cos_7 = cos_2.unsqueeze(1)
        sin_7 = sin_2.unsqueeze(1)
        q_rot_4 = query_states_4[(Ellipsis, slice(None, 64, None))]
        q_pass_4 = query_states_4[(Ellipsis, slice(64, None, None))]
        query_states_4 = None
        k_rot_4 = key_states_4[(Ellipsis, slice(None, 64, None))]
        k_pass_4 = key_states_4[(Ellipsis, slice(64, None, None))]
        key_states_4 = None
        mul_19 = q_rot_4 * cos_7
        x1_8 = q_rot_4[(Ellipsis, slice(None, 32, None))]
        x2_8 = q_rot_4[(Ellipsis, slice(32, None, None))]
        q_rot_4 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_17 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_20 = cat_17 * sin_7
        cat_17 = None
        q_embed_8 = mul_19 + mul_20
        mul_19 = mul_20 = None
        mul_21 = k_rot_4 * cos_7
        cos_7 = None
        x1_9 = k_rot_4[(Ellipsis, slice(None, 32, None))]
        x2_9 = k_rot_4[(Ellipsis, slice(32, None, None))]
        k_rot_4 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_18 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_22 = cat_18 * sin_7
        cat_18 = sin_7 = None
        k_embed_8 = mul_21 + mul_22
        mul_21 = mul_22 = None
        q_embed_9 = torch.cat([q_embed_8, q_pass_4], dim=-1)
        q_embed_8 = q_pass_4 = None
        k_embed_9 = torch.cat([k_embed_8, k_pass_4], dim=-1)
        k_embed_8 = k_pass_4 = None
        attention_mask_5 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_4 = q_embed_9.contiguous()
        q_embed_9 = None
        key_4 = k_embed_9.contiguous()
        value_4 = value_states_4.contiguous()
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = attention_mask_5 = None
        transpose_10 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_10.contiguous()
        transpose_10 = None
        reshape_4 = attn_output_21.reshape(1, 2, -1)
        attn_output_21 = None
        attn_output_22 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_22 = l_self_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_24 = torch.nn.functional.dropout(attn_output_23, 0.0, False, False)
        attn_output_23 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            hidden_states_16,
            (2048,),
            l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_17 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_9 = l_self_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_18 = torch._C._nn.gelu(hidden_states_17)
        hidden_states_17 = None
        hidden_states_19 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_18 = l_self_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_4 = torch.nn.functional.dropout(hidden_states_19, 0.0, False, False)
        hidden_states_19 = None
        add_18 = mlp_output_4 + attn_output_24
        mlp_output_4 = attn_output_24 = None
        hidden_states_20 = add_18 + hidden_states_16
        add_18 = hidden_states_16 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            hidden_states_20,
            (2048,),
            l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_5_modules_input_layernorm_parameters_bias_
        ) = None
        linear_20 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_10 = l_self_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_6 = linear_20.view((1, 2, -1, 768))
        linear_20 = None
        qkv_5 = view_6.transpose(1, 2)
        view_6 = None
        chunk_5 = qkv_5.chunk(3, dim=-1)
        qkv_5 = None
        query_states_5 = chunk_5[0]
        key_states_5 = chunk_5[1]
        value_states_5 = chunk_5[2]
        chunk_5 = None
        cos_8 = cos_2.unsqueeze(1)
        sin_8 = sin_2.unsqueeze(1)
        q_rot_5 = query_states_5[(Ellipsis, slice(None, 64, None))]
        q_pass_5 = query_states_5[(Ellipsis, slice(64, None, None))]
        query_states_5 = None
        k_rot_5 = key_states_5[(Ellipsis, slice(None, 64, None))]
        k_pass_5 = key_states_5[(Ellipsis, slice(64, None, None))]
        key_states_5 = None
        mul_23 = q_rot_5 * cos_8
        x1_10 = q_rot_5[(Ellipsis, slice(None, 32, None))]
        x2_10 = q_rot_5[(Ellipsis, slice(32, None, None))]
        q_rot_5 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_21 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_24 = cat_21 * sin_8
        cat_21 = None
        q_embed_10 = mul_23 + mul_24
        mul_23 = mul_24 = None
        mul_25 = k_rot_5 * cos_8
        cos_8 = None
        x1_11 = k_rot_5[(Ellipsis, slice(None, 32, None))]
        x2_11 = k_rot_5[(Ellipsis, slice(32, None, None))]
        k_rot_5 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_22 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_26 = cat_22 * sin_8
        cat_22 = sin_8 = None
        k_embed_10 = mul_25 + mul_26
        mul_25 = mul_26 = None
        q_embed_11 = torch.cat([q_embed_10, q_pass_5], dim=-1)
        q_embed_10 = q_pass_5 = None
        k_embed_11 = torch.cat([k_embed_10, k_pass_5], dim=-1)
        k_embed_10 = k_pass_5 = None
        attention_mask_6 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_5 = q_embed_11.contiguous()
        q_embed_11 = None
        key_5 = k_embed_11.contiguous()
        value_5 = value_states_5.contiguous()
        attn_output_25 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = attention_mask_6 = None
        transpose_12 = attn_output_25.transpose(1, 2)
        attn_output_25 = None
        attn_output_26 = transpose_12.contiguous()
        transpose_12 = None
        reshape_5 = attn_output_26.reshape(1, 2, -1)
        attn_output_26 = None
        attn_output_27 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_28 = torch._C._nn.linear(
            attn_output_27,
            l_self_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_27 = l_self_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_29 = torch.nn.functional.dropout(attn_output_28, 0.0, False, False)
        attn_output_28 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            hidden_states_20,
            (2048,),
            l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_21 = torch._C._nn.linear(
            layer_norm_11,
            l_self_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_11 = l_self_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_22 = torch._C._nn.gelu(hidden_states_21)
        hidden_states_21 = None
        hidden_states_23 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_22 = l_self_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_5 = torch.nn.functional.dropout(hidden_states_23, 0.0, False, False)
        hidden_states_23 = None
        add_22 = mlp_output_5 + attn_output_29
        mlp_output_5 = attn_output_29 = None
        hidden_states_24 = add_22 + hidden_states_20
        add_22 = hidden_states_20 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (2048,),
            l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_6_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_6_modules_input_layernorm_parameters_bias_
        ) = None
        linear_24 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_12 = l_self_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_7 = linear_24.view((1, 2, -1, 768))
        linear_24 = None
        qkv_6 = view_7.transpose(1, 2)
        view_7 = None
        chunk_6 = qkv_6.chunk(3, dim=-1)
        qkv_6 = None
        query_states_6 = chunk_6[0]
        key_states_6 = chunk_6[1]
        value_states_6 = chunk_6[2]
        chunk_6 = None
        cos_9 = cos_2.unsqueeze(1)
        sin_9 = sin_2.unsqueeze(1)
        q_rot_6 = query_states_6[(Ellipsis, slice(None, 64, None))]
        q_pass_6 = query_states_6[(Ellipsis, slice(64, None, None))]
        query_states_6 = None
        k_rot_6 = key_states_6[(Ellipsis, slice(None, 64, None))]
        k_pass_6 = key_states_6[(Ellipsis, slice(64, None, None))]
        key_states_6 = None
        mul_27 = q_rot_6 * cos_9
        x1_12 = q_rot_6[(Ellipsis, slice(None, 32, None))]
        x2_12 = q_rot_6[(Ellipsis, slice(32, None, None))]
        q_rot_6 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_25 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_28 = cat_25 * sin_9
        cat_25 = None
        q_embed_12 = mul_27 + mul_28
        mul_27 = mul_28 = None
        mul_29 = k_rot_6 * cos_9
        cos_9 = None
        x1_13 = k_rot_6[(Ellipsis, slice(None, 32, None))]
        x2_13 = k_rot_6[(Ellipsis, slice(32, None, None))]
        k_rot_6 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_26 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_30 = cat_26 * sin_9
        cat_26 = sin_9 = None
        k_embed_12 = mul_29 + mul_30
        mul_29 = mul_30 = None
        q_embed_13 = torch.cat([q_embed_12, q_pass_6], dim=-1)
        q_embed_12 = q_pass_6 = None
        k_embed_13 = torch.cat([k_embed_12, k_pass_6], dim=-1)
        k_embed_12 = k_pass_6 = None
        attention_mask_7 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_6 = q_embed_13.contiguous()
        q_embed_13 = None
        key_6 = k_embed_13.contiguous()
        value_6 = value_states_6.contiguous()
        attn_output_30 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_6,
            value_6,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_6 = key_6 = value_6 = attention_mask_7 = None
        transpose_14 = attn_output_30.transpose(1, 2)
        attn_output_30 = None
        attn_output_31 = transpose_14.contiguous()
        transpose_14 = None
        reshape_6 = attn_output_31.reshape(1, 2, -1)
        attn_output_31 = None
        attn_output_32 = reshape_6.contiguous()
        reshape_6 = None
        attn_output_33 = torch._C._nn.linear(
            attn_output_32,
            l_self_modules_layers_modules_6_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_6_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_32 = l_self_modules_layers_modules_6_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_6_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_34 = torch.nn.functional.dropout(attn_output_33, 0.0, False, False)
        attn_output_33 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (2048,),
            l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_25 = torch._C._nn.linear(
            layer_norm_13,
            l_self_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_13 = l_self_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_26 = torch._C._nn.gelu(hidden_states_25)
        hidden_states_25 = None
        hidden_states_27 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_26 = l_self_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_6 = torch.nn.functional.dropout(hidden_states_27, 0.0, False, False)
        hidden_states_27 = None
        add_26 = mlp_output_6 + attn_output_34
        mlp_output_6 = attn_output_34 = None
        hidden_states_28 = add_26 + hidden_states_24
        add_26 = hidden_states_24 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            hidden_states_28,
            (2048,),
            l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_7_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_7_modules_input_layernorm_parameters_bias_
        ) = None
        linear_28 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_14 = l_self_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_8 = linear_28.view((1, 2, -1, 768))
        linear_28 = None
        qkv_7 = view_8.transpose(1, 2)
        view_8 = None
        chunk_7 = qkv_7.chunk(3, dim=-1)
        qkv_7 = None
        query_states_7 = chunk_7[0]
        key_states_7 = chunk_7[1]
        value_states_7 = chunk_7[2]
        chunk_7 = None
        cos_10 = cos_2.unsqueeze(1)
        sin_10 = sin_2.unsqueeze(1)
        q_rot_7 = query_states_7[(Ellipsis, slice(None, 64, None))]
        q_pass_7 = query_states_7[(Ellipsis, slice(64, None, None))]
        query_states_7 = None
        k_rot_7 = key_states_7[(Ellipsis, slice(None, 64, None))]
        k_pass_7 = key_states_7[(Ellipsis, slice(64, None, None))]
        key_states_7 = None
        mul_31 = q_rot_7 * cos_10
        x1_14 = q_rot_7[(Ellipsis, slice(None, 32, None))]
        x2_14 = q_rot_7[(Ellipsis, slice(32, None, None))]
        q_rot_7 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_29 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_32 = cat_29 * sin_10
        cat_29 = None
        q_embed_14 = mul_31 + mul_32
        mul_31 = mul_32 = None
        mul_33 = k_rot_7 * cos_10
        cos_10 = None
        x1_15 = k_rot_7[(Ellipsis, slice(None, 32, None))]
        x2_15 = k_rot_7[(Ellipsis, slice(32, None, None))]
        k_rot_7 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_30 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_34 = cat_30 * sin_10
        cat_30 = sin_10 = None
        k_embed_14 = mul_33 + mul_34
        mul_33 = mul_34 = None
        q_embed_15 = torch.cat([q_embed_14, q_pass_7], dim=-1)
        q_embed_14 = q_pass_7 = None
        k_embed_15 = torch.cat([k_embed_14, k_pass_7], dim=-1)
        k_embed_14 = k_pass_7 = None
        attention_mask_8 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_7 = q_embed_15.contiguous()
        q_embed_15 = None
        key_7 = k_embed_15.contiguous()
        value_7 = value_states_7.contiguous()
        attn_output_35 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_7,
            value_7,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_7 = key_7 = value_7 = attention_mask_8 = None
        transpose_16 = attn_output_35.transpose(1, 2)
        attn_output_35 = None
        attn_output_36 = transpose_16.contiguous()
        transpose_16 = None
        reshape_7 = attn_output_36.reshape(1, 2, -1)
        attn_output_36 = None
        attn_output_37 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_38 = torch._C._nn.linear(
            attn_output_37,
            l_self_modules_layers_modules_7_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_7_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_37 = l_self_modules_layers_modules_7_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_7_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_39 = torch.nn.functional.dropout(attn_output_38, 0.0, False, False)
        attn_output_38 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            hidden_states_28,
            (2048,),
            l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_29 = torch._C._nn.linear(
            layer_norm_15,
            l_self_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_15 = l_self_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_30 = torch._C._nn.gelu(hidden_states_29)
        hidden_states_29 = None
        hidden_states_31 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_30 = l_self_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_7 = torch.nn.functional.dropout(hidden_states_31, 0.0, False, False)
        hidden_states_31 = None
        add_30 = mlp_output_7 + attn_output_39
        mlp_output_7 = attn_output_39 = None
        hidden_states_32 = add_30 + hidden_states_28
        add_30 = hidden_states_28 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            hidden_states_32,
            (2048,),
            l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_8_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_8_modules_input_layernorm_parameters_bias_
        ) = None
        linear_32 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_9 = linear_32.view((1, 2, -1, 768))
        linear_32 = None
        qkv_8 = view_9.transpose(1, 2)
        view_9 = None
        chunk_8 = qkv_8.chunk(3, dim=-1)
        qkv_8 = None
        query_states_8 = chunk_8[0]
        key_states_8 = chunk_8[1]
        value_states_8 = chunk_8[2]
        chunk_8 = None
        cos_11 = cos_2.unsqueeze(1)
        sin_11 = sin_2.unsqueeze(1)
        q_rot_8 = query_states_8[(Ellipsis, slice(None, 64, None))]
        q_pass_8 = query_states_8[(Ellipsis, slice(64, None, None))]
        query_states_8 = None
        k_rot_8 = key_states_8[(Ellipsis, slice(None, 64, None))]
        k_pass_8 = key_states_8[(Ellipsis, slice(64, None, None))]
        key_states_8 = None
        mul_35 = q_rot_8 * cos_11
        x1_16 = q_rot_8[(Ellipsis, slice(None, 32, None))]
        x2_16 = q_rot_8[(Ellipsis, slice(32, None, None))]
        q_rot_8 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_33 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_36 = cat_33 * sin_11
        cat_33 = None
        q_embed_16 = mul_35 + mul_36
        mul_35 = mul_36 = None
        mul_37 = k_rot_8 * cos_11
        cos_11 = None
        x1_17 = k_rot_8[(Ellipsis, slice(None, 32, None))]
        x2_17 = k_rot_8[(Ellipsis, slice(32, None, None))]
        k_rot_8 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_34 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_38 = cat_34 * sin_11
        cat_34 = sin_11 = None
        k_embed_16 = mul_37 + mul_38
        mul_37 = mul_38 = None
        q_embed_17 = torch.cat([q_embed_16, q_pass_8], dim=-1)
        q_embed_16 = q_pass_8 = None
        k_embed_17 = torch.cat([k_embed_16, k_pass_8], dim=-1)
        k_embed_16 = k_pass_8 = None
        attention_mask_9 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_8 = q_embed_17.contiguous()
        q_embed_17 = None
        key_8 = k_embed_17.contiguous()
        value_8 = value_states_8.contiguous()
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_8,
            value_8,
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_8 = key_8 = value_8 = attention_mask_9 = None
        transpose_18 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_18.contiguous()
        transpose_18 = None
        reshape_8 = attn_output_41.reshape(1, 2, -1)
        attn_output_41 = None
        attn_output_42 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_layers_modules_8_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_8_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_42 = l_self_modules_layers_modules_8_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_8_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_44 = torch.nn.functional.dropout(attn_output_43, 0.0, False, False)
        attn_output_43 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            hidden_states_32,
            (2048,),
            l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_33 = torch._C._nn.linear(
            layer_norm_17,
            l_self_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_17 = l_self_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_34 = torch._C._nn.gelu(hidden_states_33)
        hidden_states_33 = None
        hidden_states_35 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_34 = l_self_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_8 = torch.nn.functional.dropout(hidden_states_35, 0.0, False, False)
        hidden_states_35 = None
        add_34 = mlp_output_8 + attn_output_44
        mlp_output_8 = attn_output_44 = None
        hidden_states_36 = add_34 + hidden_states_32
        add_34 = hidden_states_32 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            hidden_states_36,
            (2048,),
            l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_9_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_9_modules_input_layernorm_parameters_bias_
        ) = None
        linear_36 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_18 = l_self_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_10 = linear_36.view((1, 2, -1, 768))
        linear_36 = None
        qkv_9 = view_10.transpose(1, 2)
        view_10 = None
        chunk_9 = qkv_9.chunk(3, dim=-1)
        qkv_9 = None
        query_states_9 = chunk_9[0]
        key_states_9 = chunk_9[1]
        value_states_9 = chunk_9[2]
        chunk_9 = None
        cos_12 = cos_2.unsqueeze(1)
        sin_12 = sin_2.unsqueeze(1)
        q_rot_9 = query_states_9[(Ellipsis, slice(None, 64, None))]
        q_pass_9 = query_states_9[(Ellipsis, slice(64, None, None))]
        query_states_9 = None
        k_rot_9 = key_states_9[(Ellipsis, slice(None, 64, None))]
        k_pass_9 = key_states_9[(Ellipsis, slice(64, None, None))]
        key_states_9 = None
        mul_39 = q_rot_9 * cos_12
        x1_18 = q_rot_9[(Ellipsis, slice(None, 32, None))]
        x2_18 = q_rot_9[(Ellipsis, slice(32, None, None))]
        q_rot_9 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_37 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_40 = cat_37 * sin_12
        cat_37 = None
        q_embed_18 = mul_39 + mul_40
        mul_39 = mul_40 = None
        mul_41 = k_rot_9 * cos_12
        cos_12 = None
        x1_19 = k_rot_9[(Ellipsis, slice(None, 32, None))]
        x2_19 = k_rot_9[(Ellipsis, slice(32, None, None))]
        k_rot_9 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_38 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_42 = cat_38 * sin_12
        cat_38 = sin_12 = None
        k_embed_18 = mul_41 + mul_42
        mul_41 = mul_42 = None
        q_embed_19 = torch.cat([q_embed_18, q_pass_9], dim=-1)
        q_embed_18 = q_pass_9 = None
        k_embed_19 = torch.cat([k_embed_18, k_pass_9], dim=-1)
        k_embed_18 = k_pass_9 = None
        attention_mask_10 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_9 = q_embed_19.contiguous()
        q_embed_19 = None
        key_9 = k_embed_19.contiguous()
        value_9 = value_states_9.contiguous()
        attn_output_45 = torch._C._nn.scaled_dot_product_attention(
            query_9,
            key_9,
            value_9,
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_9 = key_9 = value_9 = attention_mask_10 = None
        transpose_20 = attn_output_45.transpose(1, 2)
        attn_output_45 = None
        attn_output_46 = transpose_20.contiguous()
        transpose_20 = None
        reshape_9 = attn_output_46.reshape(1, 2, -1)
        attn_output_46 = None
        attn_output_47 = reshape_9.contiguous()
        reshape_9 = None
        attn_output_48 = torch._C._nn.linear(
            attn_output_47,
            l_self_modules_layers_modules_9_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_9_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_47 = l_self_modules_layers_modules_9_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_9_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_49 = torch.nn.functional.dropout(attn_output_48, 0.0, False, False)
        attn_output_48 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            hidden_states_36,
            (2048,),
            l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_37 = torch._C._nn.linear(
            layer_norm_19,
            l_self_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_19 = l_self_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_38 = torch._C._nn.gelu(hidden_states_37)
        hidden_states_37 = None
        hidden_states_39 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_38 = l_self_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_9 = torch.nn.functional.dropout(hidden_states_39, 0.0, False, False)
        hidden_states_39 = None
        add_38 = mlp_output_9 + attn_output_49
        mlp_output_9 = attn_output_49 = None
        hidden_states_40 = add_38 + hidden_states_36
        add_38 = hidden_states_36 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            hidden_states_40,
            (2048,),
            l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_10_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_10_modules_input_layernorm_parameters_bias_
        ) = None
        linear_40 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_20 = l_self_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_11 = linear_40.view((1, 2, -1, 768))
        linear_40 = None
        qkv_10 = view_11.transpose(1, 2)
        view_11 = None
        chunk_10 = qkv_10.chunk(3, dim=-1)
        qkv_10 = None
        query_states_10 = chunk_10[0]
        key_states_10 = chunk_10[1]
        value_states_10 = chunk_10[2]
        chunk_10 = None
        cos_13 = cos_2.unsqueeze(1)
        sin_13 = sin_2.unsqueeze(1)
        q_rot_10 = query_states_10[(Ellipsis, slice(None, 64, None))]
        q_pass_10 = query_states_10[(Ellipsis, slice(64, None, None))]
        query_states_10 = None
        k_rot_10 = key_states_10[(Ellipsis, slice(None, 64, None))]
        k_pass_10 = key_states_10[(Ellipsis, slice(64, None, None))]
        key_states_10 = None
        mul_43 = q_rot_10 * cos_13
        x1_20 = q_rot_10[(Ellipsis, slice(None, 32, None))]
        x2_20 = q_rot_10[(Ellipsis, slice(32, None, None))]
        q_rot_10 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_41 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_44 = cat_41 * sin_13
        cat_41 = None
        q_embed_20 = mul_43 + mul_44
        mul_43 = mul_44 = None
        mul_45 = k_rot_10 * cos_13
        cos_13 = None
        x1_21 = k_rot_10[(Ellipsis, slice(None, 32, None))]
        x2_21 = k_rot_10[(Ellipsis, slice(32, None, None))]
        k_rot_10 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_42 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_46 = cat_42 * sin_13
        cat_42 = sin_13 = None
        k_embed_20 = mul_45 + mul_46
        mul_45 = mul_46 = None
        q_embed_21 = torch.cat([q_embed_20, q_pass_10], dim=-1)
        q_embed_20 = q_pass_10 = None
        k_embed_21 = torch.cat([k_embed_20, k_pass_10], dim=-1)
        k_embed_20 = k_pass_10 = None
        attention_mask_11 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_10 = q_embed_21.contiguous()
        q_embed_21 = None
        key_10 = k_embed_21.contiguous()
        value_10 = value_states_10.contiguous()
        attn_output_50 = torch._C._nn.scaled_dot_product_attention(
            query_10,
            key_10,
            value_10,
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_10 = key_10 = value_10 = attention_mask_11 = None
        transpose_22 = attn_output_50.transpose(1, 2)
        attn_output_50 = None
        attn_output_51 = transpose_22.contiguous()
        transpose_22 = None
        reshape_10 = attn_output_51.reshape(1, 2, -1)
        attn_output_51 = None
        attn_output_52 = reshape_10.contiguous()
        reshape_10 = None
        attn_output_53 = torch._C._nn.linear(
            attn_output_52,
            l_self_modules_layers_modules_10_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_10_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_52 = l_self_modules_layers_modules_10_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_10_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_54 = torch.nn.functional.dropout(attn_output_53, 0.0, False, False)
        attn_output_53 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            hidden_states_40,
            (2048,),
            l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_41 = torch._C._nn.linear(
            layer_norm_21,
            l_self_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_21 = l_self_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_42 = torch._C._nn.gelu(hidden_states_41)
        hidden_states_41 = None
        hidden_states_43 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_42 = l_self_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_10 = torch.nn.functional.dropout(hidden_states_43, 0.0, False, False)
        hidden_states_43 = None
        add_42 = mlp_output_10 + attn_output_54
        mlp_output_10 = attn_output_54 = None
        hidden_states_44 = add_42 + hidden_states_40
        add_42 = hidden_states_40 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            hidden_states_44,
            (2048,),
            l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_11_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_11_modules_input_layernorm_parameters_bias_
        ) = None
        linear_44 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_22 = l_self_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_12 = linear_44.view((1, 2, -1, 768))
        linear_44 = None
        qkv_11 = view_12.transpose(1, 2)
        view_12 = None
        chunk_11 = qkv_11.chunk(3, dim=-1)
        qkv_11 = None
        query_states_11 = chunk_11[0]
        key_states_11 = chunk_11[1]
        value_states_11 = chunk_11[2]
        chunk_11 = None
        cos_14 = cos_2.unsqueeze(1)
        sin_14 = sin_2.unsqueeze(1)
        q_rot_11 = query_states_11[(Ellipsis, slice(None, 64, None))]
        q_pass_11 = query_states_11[(Ellipsis, slice(64, None, None))]
        query_states_11 = None
        k_rot_11 = key_states_11[(Ellipsis, slice(None, 64, None))]
        k_pass_11 = key_states_11[(Ellipsis, slice(64, None, None))]
        key_states_11 = None
        mul_47 = q_rot_11 * cos_14
        x1_22 = q_rot_11[(Ellipsis, slice(None, 32, None))]
        x2_22 = q_rot_11[(Ellipsis, slice(32, None, None))]
        q_rot_11 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_45 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_48 = cat_45 * sin_14
        cat_45 = None
        q_embed_22 = mul_47 + mul_48
        mul_47 = mul_48 = None
        mul_49 = k_rot_11 * cos_14
        cos_14 = None
        x1_23 = k_rot_11[(Ellipsis, slice(None, 32, None))]
        x2_23 = k_rot_11[(Ellipsis, slice(32, None, None))]
        k_rot_11 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_46 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_50 = cat_46 * sin_14
        cat_46 = sin_14 = None
        k_embed_22 = mul_49 + mul_50
        mul_49 = mul_50 = None
        q_embed_23 = torch.cat([q_embed_22, q_pass_11], dim=-1)
        q_embed_22 = q_pass_11 = None
        k_embed_23 = torch.cat([k_embed_22, k_pass_11], dim=-1)
        k_embed_22 = k_pass_11 = None
        attention_mask_12 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_11 = q_embed_23.contiguous()
        q_embed_23 = None
        key_11 = k_embed_23.contiguous()
        value_11 = value_states_11.contiguous()
        attn_output_55 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_11,
            value_11,
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_11 = key_11 = value_11 = attention_mask_12 = None
        transpose_24 = attn_output_55.transpose(1, 2)
        attn_output_55 = None
        attn_output_56 = transpose_24.contiguous()
        transpose_24 = None
        reshape_11 = attn_output_56.reshape(1, 2, -1)
        attn_output_56 = None
        attn_output_57 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_58 = torch._C._nn.linear(
            attn_output_57,
            l_self_modules_layers_modules_11_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_11_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_57 = l_self_modules_layers_modules_11_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_11_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_59 = torch.nn.functional.dropout(attn_output_58, 0.0, False, False)
        attn_output_58 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            hidden_states_44,
            (2048,),
            l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_45 = torch._C._nn.linear(
            layer_norm_23,
            l_self_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_23 = l_self_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_46 = torch._C._nn.gelu(hidden_states_45)
        hidden_states_45 = None
        hidden_states_47 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_46 = l_self_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_11 = torch.nn.functional.dropout(hidden_states_47, 0.0, False, False)
        hidden_states_47 = None
        add_46 = mlp_output_11 + attn_output_59
        mlp_output_11 = attn_output_59 = None
        hidden_states_48 = add_46 + hidden_states_44
        add_46 = hidden_states_44 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            hidden_states_48,
            (2048,),
            l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_12_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_12_modules_input_layernorm_parameters_bias_
        ) = None
        linear_48 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_24 = l_self_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_13 = linear_48.view((1, 2, -1, 768))
        linear_48 = None
        qkv_12 = view_13.transpose(1, 2)
        view_13 = None
        chunk_12 = qkv_12.chunk(3, dim=-1)
        qkv_12 = None
        query_states_12 = chunk_12[0]
        key_states_12 = chunk_12[1]
        value_states_12 = chunk_12[2]
        chunk_12 = None
        cos_15 = cos_2.unsqueeze(1)
        sin_15 = sin_2.unsqueeze(1)
        q_rot_12 = query_states_12[(Ellipsis, slice(None, 64, None))]
        q_pass_12 = query_states_12[(Ellipsis, slice(64, None, None))]
        query_states_12 = None
        k_rot_12 = key_states_12[(Ellipsis, slice(None, 64, None))]
        k_pass_12 = key_states_12[(Ellipsis, slice(64, None, None))]
        key_states_12 = None
        mul_51 = q_rot_12 * cos_15
        x1_24 = q_rot_12[(Ellipsis, slice(None, 32, None))]
        x2_24 = q_rot_12[(Ellipsis, slice(32, None, None))]
        q_rot_12 = None
        neg_24 = -x2_24
        x2_24 = None
        cat_49 = torch.cat((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        mul_52 = cat_49 * sin_15
        cat_49 = None
        q_embed_24 = mul_51 + mul_52
        mul_51 = mul_52 = None
        mul_53 = k_rot_12 * cos_15
        cos_15 = None
        x1_25 = k_rot_12[(Ellipsis, slice(None, 32, None))]
        x2_25 = k_rot_12[(Ellipsis, slice(32, None, None))]
        k_rot_12 = None
        neg_25 = -x2_25
        x2_25 = None
        cat_50 = torch.cat((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        mul_54 = cat_50 * sin_15
        cat_50 = sin_15 = None
        k_embed_24 = mul_53 + mul_54
        mul_53 = mul_54 = None
        q_embed_25 = torch.cat([q_embed_24, q_pass_12], dim=-1)
        q_embed_24 = q_pass_12 = None
        k_embed_25 = torch.cat([k_embed_24, k_pass_12], dim=-1)
        k_embed_24 = k_pass_12 = None
        attention_mask_13 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_12 = q_embed_25.contiguous()
        q_embed_25 = None
        key_12 = k_embed_25.contiguous()
        value_12 = value_states_12.contiguous()
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_12,
            key_12,
            value_12,
            attn_mask=attention_mask_13,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_12 = key_12 = value_12 = attention_mask_13 = None
        transpose_26 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_26.contiguous()
        transpose_26 = None
        reshape_12 = attn_output_61.reshape(1, 2, -1)
        attn_output_61 = None
        attn_output_62 = reshape_12.contiguous()
        reshape_12 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_layers_modules_12_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_12_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_62 = l_self_modules_layers_modules_12_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_12_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_64 = torch.nn.functional.dropout(attn_output_63, 0.0, False, False)
        attn_output_63 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            hidden_states_48,
            (2048,),
            l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_49 = torch._C._nn.linear(
            layer_norm_25,
            l_self_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_25 = l_self_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_50 = torch._C._nn.gelu(hidden_states_49)
        hidden_states_49 = None
        hidden_states_51 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_50 = l_self_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_12 = torch.nn.functional.dropout(hidden_states_51, 0.0, False, False)
        hidden_states_51 = None
        add_50 = mlp_output_12 + attn_output_64
        mlp_output_12 = attn_output_64 = None
        hidden_states_52 = add_50 + hidden_states_48
        add_50 = hidden_states_48 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            hidden_states_52,
            (2048,),
            l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_13_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_13_modules_input_layernorm_parameters_bias_
        ) = None
        linear_52 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_26 = l_self_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_14 = linear_52.view((1, 2, -1, 768))
        linear_52 = None
        qkv_13 = view_14.transpose(1, 2)
        view_14 = None
        chunk_13 = qkv_13.chunk(3, dim=-1)
        qkv_13 = None
        query_states_13 = chunk_13[0]
        key_states_13 = chunk_13[1]
        value_states_13 = chunk_13[2]
        chunk_13 = None
        cos_16 = cos_2.unsqueeze(1)
        sin_16 = sin_2.unsqueeze(1)
        q_rot_13 = query_states_13[(Ellipsis, slice(None, 64, None))]
        q_pass_13 = query_states_13[(Ellipsis, slice(64, None, None))]
        query_states_13 = None
        k_rot_13 = key_states_13[(Ellipsis, slice(None, 64, None))]
        k_pass_13 = key_states_13[(Ellipsis, slice(64, None, None))]
        key_states_13 = None
        mul_55 = q_rot_13 * cos_16
        x1_26 = q_rot_13[(Ellipsis, slice(None, 32, None))]
        x2_26 = q_rot_13[(Ellipsis, slice(32, None, None))]
        q_rot_13 = None
        neg_26 = -x2_26
        x2_26 = None
        cat_53 = torch.cat((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        mul_56 = cat_53 * sin_16
        cat_53 = None
        q_embed_26 = mul_55 + mul_56
        mul_55 = mul_56 = None
        mul_57 = k_rot_13 * cos_16
        cos_16 = None
        x1_27 = k_rot_13[(Ellipsis, slice(None, 32, None))]
        x2_27 = k_rot_13[(Ellipsis, slice(32, None, None))]
        k_rot_13 = None
        neg_27 = -x2_27
        x2_27 = None
        cat_54 = torch.cat((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        mul_58 = cat_54 * sin_16
        cat_54 = sin_16 = None
        k_embed_26 = mul_57 + mul_58
        mul_57 = mul_58 = None
        q_embed_27 = torch.cat([q_embed_26, q_pass_13], dim=-1)
        q_embed_26 = q_pass_13 = None
        k_embed_27 = torch.cat([k_embed_26, k_pass_13], dim=-1)
        k_embed_26 = k_pass_13 = None
        attention_mask_14 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_13 = q_embed_27.contiguous()
        q_embed_27 = None
        key_13 = k_embed_27.contiguous()
        value_13 = value_states_13.contiguous()
        attn_output_65 = torch._C._nn.scaled_dot_product_attention(
            query_13,
            key_13,
            value_13,
            attn_mask=attention_mask_14,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_13 = key_13 = value_13 = attention_mask_14 = None
        transpose_28 = attn_output_65.transpose(1, 2)
        attn_output_65 = None
        attn_output_66 = transpose_28.contiguous()
        transpose_28 = None
        reshape_13 = attn_output_66.reshape(1, 2, -1)
        attn_output_66 = None
        attn_output_67 = reshape_13.contiguous()
        reshape_13 = None
        attn_output_68 = torch._C._nn.linear(
            attn_output_67,
            l_self_modules_layers_modules_13_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_13_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_67 = l_self_modules_layers_modules_13_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_13_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_69 = torch.nn.functional.dropout(attn_output_68, 0.0, False, False)
        attn_output_68 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            hidden_states_52,
            (2048,),
            l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_53 = torch._C._nn.linear(
            layer_norm_27,
            l_self_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_27 = l_self_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_54 = torch._C._nn.gelu(hidden_states_53)
        hidden_states_53 = None
        hidden_states_55 = torch._C._nn.linear(
            hidden_states_54,
            l_self_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_54 = l_self_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_13 = torch.nn.functional.dropout(hidden_states_55, 0.0, False, False)
        hidden_states_55 = None
        add_54 = mlp_output_13 + attn_output_69
        mlp_output_13 = attn_output_69 = None
        hidden_states_56 = add_54 + hidden_states_52
        add_54 = hidden_states_52 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            hidden_states_56,
            (2048,),
            l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_14_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_14_modules_input_layernorm_parameters_bias_
        ) = None
        linear_56 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_28 = l_self_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_15 = linear_56.view((1, 2, -1, 768))
        linear_56 = None
        qkv_14 = view_15.transpose(1, 2)
        view_15 = None
        chunk_14 = qkv_14.chunk(3, dim=-1)
        qkv_14 = None
        query_states_14 = chunk_14[0]
        key_states_14 = chunk_14[1]
        value_states_14 = chunk_14[2]
        chunk_14 = None
        cos_17 = cos_2.unsqueeze(1)
        sin_17 = sin_2.unsqueeze(1)
        q_rot_14 = query_states_14[(Ellipsis, slice(None, 64, None))]
        q_pass_14 = query_states_14[(Ellipsis, slice(64, None, None))]
        query_states_14 = None
        k_rot_14 = key_states_14[(Ellipsis, slice(None, 64, None))]
        k_pass_14 = key_states_14[(Ellipsis, slice(64, None, None))]
        key_states_14 = None
        mul_59 = q_rot_14 * cos_17
        x1_28 = q_rot_14[(Ellipsis, slice(None, 32, None))]
        x2_28 = q_rot_14[(Ellipsis, slice(32, None, None))]
        q_rot_14 = None
        neg_28 = -x2_28
        x2_28 = None
        cat_57 = torch.cat((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        mul_60 = cat_57 * sin_17
        cat_57 = None
        q_embed_28 = mul_59 + mul_60
        mul_59 = mul_60 = None
        mul_61 = k_rot_14 * cos_17
        cos_17 = None
        x1_29 = k_rot_14[(Ellipsis, slice(None, 32, None))]
        x2_29 = k_rot_14[(Ellipsis, slice(32, None, None))]
        k_rot_14 = None
        neg_29 = -x2_29
        x2_29 = None
        cat_58 = torch.cat((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        mul_62 = cat_58 * sin_17
        cat_58 = sin_17 = None
        k_embed_28 = mul_61 + mul_62
        mul_61 = mul_62 = None
        q_embed_29 = torch.cat([q_embed_28, q_pass_14], dim=-1)
        q_embed_28 = q_pass_14 = None
        k_embed_29 = torch.cat([k_embed_28, k_pass_14], dim=-1)
        k_embed_28 = k_pass_14 = None
        attention_mask_15 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_14 = q_embed_29.contiguous()
        q_embed_29 = None
        key_14 = k_embed_29.contiguous()
        value_14 = value_states_14.contiguous()
        attn_output_70 = torch._C._nn.scaled_dot_product_attention(
            query_14,
            key_14,
            value_14,
            attn_mask=attention_mask_15,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_14 = key_14 = value_14 = attention_mask_15 = None
        transpose_30 = attn_output_70.transpose(1, 2)
        attn_output_70 = None
        attn_output_71 = transpose_30.contiguous()
        transpose_30 = None
        reshape_14 = attn_output_71.reshape(1, 2, -1)
        attn_output_71 = None
        attn_output_72 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_73 = torch._C._nn.linear(
            attn_output_72,
            l_self_modules_layers_modules_14_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_14_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_72 = l_self_modules_layers_modules_14_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_14_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_74 = torch.nn.functional.dropout(attn_output_73, 0.0, False, False)
        attn_output_73 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            hidden_states_56,
            (2048,),
            l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_57 = torch._C._nn.linear(
            layer_norm_29,
            l_self_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_29 = l_self_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_58 = torch._C._nn.gelu(hidden_states_57)
        hidden_states_57 = None
        hidden_states_59 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_58 = l_self_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_14 = torch.nn.functional.dropout(hidden_states_59, 0.0, False, False)
        hidden_states_59 = None
        add_58 = mlp_output_14 + attn_output_74
        mlp_output_14 = attn_output_74 = None
        hidden_states_60 = add_58 + hidden_states_56
        add_58 = hidden_states_56 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            hidden_states_60,
            (2048,),
            l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_,
            l_self_modules_layers_modules_15_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_layers_modules_15_modules_input_layernorm_parameters_bias_
        ) = None
        linear_60 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_30 = l_self_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_16 = linear_60.view((1, 2, -1, 768))
        linear_60 = None
        qkv_15 = view_16.transpose(1, 2)
        view_16 = None
        chunk_15 = qkv_15.chunk(3, dim=-1)
        qkv_15 = None
        query_states_15 = chunk_15[0]
        key_states_15 = chunk_15[1]
        value_states_15 = chunk_15[2]
        chunk_15 = None
        cos_18 = cos_2.unsqueeze(1)
        cos_2 = None
        sin_18 = sin_2.unsqueeze(1)
        sin_2 = None
        q_rot_15 = query_states_15[(Ellipsis, slice(None, 64, None))]
        q_pass_15 = query_states_15[(Ellipsis, slice(64, None, None))]
        query_states_15 = None
        k_rot_15 = key_states_15[(Ellipsis, slice(None, 64, None))]
        k_pass_15 = key_states_15[(Ellipsis, slice(64, None, None))]
        key_states_15 = None
        mul_63 = q_rot_15 * cos_18
        x1_30 = q_rot_15[(Ellipsis, slice(None, 32, None))]
        x2_30 = q_rot_15[(Ellipsis, slice(32, None, None))]
        q_rot_15 = None
        neg_30 = -x2_30
        x2_30 = None
        cat_61 = torch.cat((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        mul_64 = cat_61 * sin_18
        cat_61 = None
        q_embed_30 = mul_63 + mul_64
        mul_63 = mul_64 = None
        mul_65 = k_rot_15 * cos_18
        cos_18 = None
        x1_31 = k_rot_15[(Ellipsis, slice(None, 32, None))]
        x2_31 = k_rot_15[(Ellipsis, slice(32, None, None))]
        k_rot_15 = None
        neg_31 = -x2_31
        x2_31 = None
        cat_62 = torch.cat((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        mul_66 = cat_62 * sin_18
        cat_62 = sin_18 = None
        k_embed_30 = mul_65 + mul_66
        mul_65 = mul_66 = None
        q_embed_31 = torch.cat([q_embed_30, q_pass_15], dim=-1)
        q_embed_30 = q_pass_15 = None
        k_embed_31 = torch.cat([k_embed_30, k_pass_15], dim=-1)
        k_embed_30 = k_pass_15 = None
        attention_mask_16 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        causal_mask_2 = None
        query_15 = q_embed_31.contiguous()
        q_embed_31 = None
        key_15 = k_embed_31.contiguous()
        value_15 = value_states_15.contiguous()
        attn_output_75 = torch._C._nn.scaled_dot_product_attention(
            query_15,
            key_15,
            value_15,
            attn_mask=attention_mask_16,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_15 = key_15 = value_15 = attention_mask_16 = None
        transpose_32 = attn_output_75.transpose(1, 2)
        attn_output_75 = None
        attn_output_76 = transpose_32.contiguous()
        transpose_32 = None
        reshape_15 = attn_output_76.reshape(1, 2, -1)
        attn_output_76 = None
        attn_output_77 = reshape_15.contiguous()
        reshape_15 = None
        attn_output_78 = torch._C._nn.linear(
            attn_output_77,
            l_self_modules_layers_modules_15_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_layers_modules_15_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_77 = l_self_modules_layers_modules_15_modules_attention_modules_dense_parameters_weight_ = l_self_modules_layers_modules_15_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_79 = torch.nn.functional.dropout(attn_output_78, 0.0, False, False)
        attn_output_78 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            hidden_states_60,
            (2048,),
            l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_61 = torch._C._nn.linear(
            layer_norm_31,
            l_self_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_31 = l_self_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_62 = torch._C._nn.gelu(hidden_states_61)
        hidden_states_61 = None
        hidden_states_63 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_62 = l_self_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_15 = torch.nn.functional.dropout(hidden_states_63, 0.0, False, False)
        hidden_states_63 = None
        add_62 = mlp_output_15 + attn_output_79
        mlp_output_15 = attn_output_79 = None
        hidden_states_64 = add_62 + hidden_states_60
        add_62 = hidden_states_60 = None
        hidden_states_65 = torch.nn.functional.layer_norm(
            hidden_states_64,
            (2048,),
            l_self_modules_final_layer_norm_parameters_weight_,
            l_self_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_64 = (
            l_self_modules_final_layer_norm_parameters_weight_
        ) = l_self_modules_final_layer_norm_parameters_bias_ = None
        return (
            value_states,
            k_embed_1,
            value_states_1,
            k_embed_3,
            value_states_2,
            k_embed_5,
            value_states_3,
            k_embed_7,
            value_states_4,
            k_embed_9,
            value_states_5,
            k_embed_11,
            value_states_6,
            k_embed_13,
            value_states_7,
            k_embed_15,
            value_states_8,
            k_embed_17,
            value_states_9,
            k_embed_19,
            value_states_10,
            k_embed_21,
            value_states_11,
            k_embed_23,
            value_states_12,
            k_embed_25,
            value_states_13,
            k_embed_27,
            value_states_14,
            k_embed_29,
            value_states_15,
            k_embed_31,
            hidden_states_65,
        )
