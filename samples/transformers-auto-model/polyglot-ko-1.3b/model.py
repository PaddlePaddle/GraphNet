import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_kwargs_input_ids_: torch.Tensor,
        L_self_modules_gpt_neox_modules_embed_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_kwargs_attention_mask_: torch.Tensor,
        L_self_modules_gpt_neox_modules_rotary_emb_buffers_inv_freq_: torch.Tensor,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_6_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_6_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_7_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_7_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_8_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_8_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_9_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_9_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_10_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_10_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_11_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_11_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_12_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_12_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_13_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_13_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_14_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_14_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_15_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_15_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_16_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_16_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_16_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_17_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_17_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_17_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_18_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_18_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_18_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_19_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_19_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_19_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_20_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_20_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_20_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_21_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_21_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_21_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_22_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_22_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_22_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_23_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_23_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_23_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_gpt_neox_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_embed_out_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_kwargs_input_ids_ = L_kwargs_input_ids_
        l_self_modules_gpt_neox_modules_embed_in_parameters_weight_ = (
            L_self_modules_gpt_neox_modules_embed_in_parameters_weight_
        )
        l_kwargs_attention_mask_ = L_kwargs_attention_mask_
        l_self_modules_gpt_neox_modules_rotary_emb_buffers_inv_freq_ = (
            L_self_modules_gpt_neox_modules_rotary_emb_buffers_inv_freq_
        )
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_6_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_6_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_7_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_7_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_8_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_8_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_9_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_9_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_10_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_10_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_11_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_11_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_12_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_12_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_13_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_13_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_14_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_14_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_15_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_15_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_16_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_16_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_16_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_16_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_16_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_17_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_17_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_17_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_17_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_17_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_18_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_18_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_18_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_18_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_18_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_19_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_19_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_19_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_19_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_19_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_20_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_20_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_20_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_20_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_20_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_21_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_21_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_21_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_21_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_21_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_22_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_22_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_22_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_22_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_22_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_23_modules_input_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_23_modules_input_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_23_modules_input_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_query_key_value_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_query_key_value_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_query_key_value_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_query_key_value_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_dense_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_dense_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_dense_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_dense_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_23_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_23_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_gpt_neox_modules_final_layer_norm_parameters_weight_ = (
            L_self_modules_gpt_neox_modules_final_layer_norm_parameters_weight_
        )
        l_self_modules_gpt_neox_modules_final_layer_norm_parameters_bias_ = (
            L_self_modules_gpt_neox_modules_final_layer_norm_parameters_bias_
        )
        l_self_modules_embed_out_parameters_weight_ = (
            L_self_modules_embed_out_parameters_weight_
        )
        inputs_embeds = torch.nn.functional.embedding(
            l_kwargs_input_ids_,
            l_self_modules_gpt_neox_modules_embed_in_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        l_kwargs_input_ids_ = (
            l_self_modules_gpt_neox_modules_embed_in_parameters_weight_
        ) = None
        cache_position = torch.arange(0, 45, device=device(type="cuda", index=0))
        position_ids = cache_position.unsqueeze(0)
        attention_mask = l_kwargs_attention_mask_.to(
            device=device(type="cuda", index=0), dtype=torch.bool
        )
        l_kwargs_attention_mask_ = None
        kv_arange = torch.arange(45, device=device(type="cuda", index=0))
        kv_arange += 0
        kv_arange_1 = kv_arange
        kv_arange = None
        batch_arange = torch.arange(1, device=device(type="cuda", index=0))
        head_arange = torch.arange(1, device=device(type="cuda", index=0))
        lazy_load_decompositions = (
            torch._functorch.predispatch.lazy_load_decompositions()
        )
        lazy_load_decompositions = None
        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting = None
        child = torch._functorch.predispatch._add_batch_dim(batch_arange, 0, 1)
        batch_arange = None
        lazy_load_decompositions_1 = (
            torch._functorch.predispatch.lazy_load_decompositions()
        )
        lazy_load_decompositions_1 = None
        _vmap_increment_nesting_1 = (
            torch._functorch.predispatch._vmap_increment_nesting(1, "error")
        )
        _vmap_increment_nesting_1 = None
        child_1 = torch._functorch.predispatch._add_batch_dim(head_arange, 0, 2)
        head_arange = child_1 = None
        lazy_load_decompositions_2 = (
            torch._functorch.predispatch.lazy_load_decompositions()
        )
        lazy_load_decompositions_2 = None
        _vmap_increment_nesting_2 = (
            torch._functorch.predispatch._vmap_increment_nesting(45, "error")
        )
        _vmap_increment_nesting_2 = None
        child_2 = torch._functorch.predispatch._add_batch_dim(cache_position, 0, 3)
        cache_position = None
        lazy_load_decompositions_3 = (
            torch._functorch.predispatch.lazy_load_decompositions()
        )
        lazy_load_decompositions_3 = None
        _vmap_increment_nesting_3 = (
            torch._functorch.predispatch._vmap_increment_nesting(45, "error")
        )
        _vmap_increment_nesting_3 = None
        child_3 = torch._functorch.predispatch._add_batch_dim(kv_arange_1, 0, 4)
        kv_arange_1 = None
        result = child_2.new_ones((), dtype=torch.bool)
        le = child_3.le(child_2)
        child_2 = None
        to_1 = le.to(device(type="cuda", index=0))
        le = None
        result_1 = result.__and__(to_1)
        result = to_1 = None
        index = torch.ops.aten.index(attention_mask, [child, child_3])
        attention_mask = child = child_3 = None
        to_2 = index.to(device(type="cuda", index=0))
        index = None
        result_2 = result_1.__and__(to_2)
        result_1 = to_2 = None
        batched_outputs = torch._functorch.predispatch._remove_batch_dim(
            result_2, 4, 45, 0
        )
        result_2 = None
        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting()
        _vmap_decrement_nesting = None
        batched_outputs_1 = torch._functorch.predispatch._remove_batch_dim(
            batched_outputs, 3, 45, 0
        )
        batched_outputs = None
        _vmap_decrement_nesting_1 = (
            torch._functorch.predispatch._vmap_decrement_nesting()
        )
        _vmap_decrement_nesting_1 = None
        batched_outputs_2 = torch._functorch.predispatch._remove_batch_dim(
            batched_outputs_1, 2, 1, 0
        )
        batched_outputs_1 = None
        _vmap_decrement_nesting_2 = (
            torch._functorch.predispatch._vmap_decrement_nesting()
        )
        _vmap_decrement_nesting_2 = None
        causal_mask = torch._functorch.predispatch._remove_batch_dim(
            batched_outputs_2, 1, 1, 0
        )
        batched_outputs_2 = None
        _vmap_decrement_nesting_3 = (
            torch._functorch.predispatch._vmap_decrement_nesting()
        )
        _vmap_decrement_nesting_3 = None
        hidden_states = torch.nn.functional.dropout(inputs_embeds, 0.0, False, False)
        inputs_embeds = None
        getitem = l_self_modules_gpt_neox_modules_rotary_emb_buffers_inv_freq_[
            (None, slice(None, None, None), None)
        ]
        l_self_modules_gpt_neox_modules_rotary_emb_buffers_inv_freq_ = None
        float_1 = getitem.float()
        getitem = None
        expand = float_1.expand(1, -1, 1)
        float_1 = None
        inv_freq_expanded = expand.to(device(type="cuda", index=0))
        expand = None
        getitem_1 = position_ids[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        position_ids = None
        position_ids_expanded = getitem_1.float()
        getitem_1 = None
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
        layer_norm = torch.nn.functional.layer_norm(
            hidden_states,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_bias_ = (None)
        linear = torch._C._nn.linear(
            layer_norm,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm = l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view = linear.view((1, 45, -1, 384))
        linear = None
        qkv = view.transpose(1, 2)
        view = None
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
        mul_2 = q_rot * cos_3
        x1 = q_rot[(Ellipsis, slice(None, 32, None))]
        x2 = q_rot[(Ellipsis, slice(32, None, None))]
        q_rot = None
        neg = -x2
        x2 = None
        cat_1 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_3 = cat_1 * sin_3
        cat_1 = None
        q_embed = mul_2 + mul_3
        mul_2 = mul_3 = None
        mul_4 = k_rot * cos_3
        cos_3 = None
        x1_1 = k_rot[(Ellipsis, slice(None, 32, None))]
        x2_1 = k_rot[(Ellipsis, slice(32, None, None))]
        k_rot = None
        neg_1 = -x2_1
        x2_1 = None
        cat_2 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_5 = cat_2 * sin_3
        cat_2 = sin_3 = None
        k_embed = mul_4 + mul_5
        mul_4 = mul_5 = None
        q_embed_1 = torch.cat([q_embed, q_pass], dim=-1)
        q_embed = q_pass = None
        k_embed_1 = torch.cat([k_embed, k_pass], dim=-1)
        k_embed = k_pass = None
        attention_mask_1 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output = torch._C._nn.scaled_dot_product_attention(
            q_embed_1,
            k_embed_1,
            value_states,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_1 = k_embed_1 = value_states = attention_mask_1 = None
        transpose_2 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_2.contiguous()
        transpose_2 = None
        reshape = attn_output_1.reshape(1, 45, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_2 = l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_4 = torch.nn.functional.dropout(attn_output_3, 0.0, False, False)
        attn_output_3 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            hidden_states,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_1 = torch._C._nn.linear(
            layer_norm_1,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_1 = l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_2 = torch._C._nn.gelu(hidden_states_1)
        hidden_states_1 = None
        hidden_states_3 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output = torch.nn.functional.dropout(hidden_states_3, 0.0, False, False)
        hidden_states_3 = None
        add_2 = mlp_output + attn_output_4
        mlp_output = attn_output_4 = None
        hidden_states_4 = add_2 + hidden_states
        add_2 = hidden_states = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_bias_ = (None)
        linear_4 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_2 = l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_1 = linear_4.view((1, 45, -1, 384))
        linear_4 = None
        qkv_1 = view_1.transpose(1, 2)
        view_1 = None
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
        mul_6 = q_rot_1 * cos_4
        x1_2 = q_rot_1[(Ellipsis, slice(None, 32, None))]
        x2_2 = q_rot_1[(Ellipsis, slice(32, None, None))]
        q_rot_1 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_5 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_7 = cat_5 * sin_4
        cat_5 = None
        q_embed_2 = mul_6 + mul_7
        mul_6 = mul_7 = None
        mul_8 = k_rot_1 * cos_4
        cos_4 = None
        x1_3 = k_rot_1[(Ellipsis, slice(None, 32, None))]
        x2_3 = k_rot_1[(Ellipsis, slice(32, None, None))]
        k_rot_1 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_6 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_9 = cat_6 * sin_4
        cat_6 = sin_4 = None
        k_embed_2 = mul_8 + mul_9
        mul_8 = mul_9 = None
        q_embed_3 = torch.cat([q_embed_2, q_pass_1], dim=-1)
        q_embed_2 = q_pass_1 = None
        k_embed_3 = torch.cat([k_embed_2, k_pass_1], dim=-1)
        k_embed_2 = k_pass_1 = None
        attention_mask_2 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_5 = torch._C._nn.scaled_dot_product_attention(
            q_embed_3,
            k_embed_3,
            value_states_1,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_3 = k_embed_3 = value_states_1 = attention_mask_2 = None
        transpose_4 = attn_output_5.transpose(1, 2)
        attn_output_5 = None
        attn_output_6 = transpose_4.contiguous()
        transpose_4 = None
        reshape_1 = attn_output_6.reshape(1, 45, -1)
        attn_output_6 = None
        attn_output_7 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_8 = torch._C._nn.linear(
            attn_output_7,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_7 = l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_9 = torch.nn.functional.dropout(attn_output_8, 0.0, False, False)
        attn_output_8 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_5 = torch._C._nn.linear(
            layer_norm_3,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_3 = l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_6 = torch._C._nn.gelu(hidden_states_5)
        hidden_states_5 = None
        hidden_states_7 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_6 = l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_1 = torch.nn.functional.dropout(hidden_states_7, 0.0, False, False)
        hidden_states_7 = None
        add_6 = mlp_output_1 + attn_output_9
        mlp_output_1 = attn_output_9 = None
        hidden_states_8 = add_6 + hidden_states_4
        add_6 = hidden_states_4 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            hidden_states_8,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_bias_ = (None)
        linear_8 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_4 = l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_2 = linear_8.view((1, 45, -1, 384))
        linear_8 = None
        qkv_2 = view_2.transpose(1, 2)
        view_2 = None
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
        mul_10 = q_rot_2 * cos_5
        x1_4 = q_rot_2[(Ellipsis, slice(None, 32, None))]
        x2_4 = q_rot_2[(Ellipsis, slice(32, None, None))]
        q_rot_2 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_9 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_11 = cat_9 * sin_5
        cat_9 = None
        q_embed_4 = mul_10 + mul_11
        mul_10 = mul_11 = None
        mul_12 = k_rot_2 * cos_5
        cos_5 = None
        x1_5 = k_rot_2[(Ellipsis, slice(None, 32, None))]
        x2_5 = k_rot_2[(Ellipsis, slice(32, None, None))]
        k_rot_2 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_10 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_13 = cat_10 * sin_5
        cat_10 = sin_5 = None
        k_embed_4 = mul_12 + mul_13
        mul_12 = mul_13 = None
        q_embed_5 = torch.cat([q_embed_4, q_pass_2], dim=-1)
        q_embed_4 = q_pass_2 = None
        k_embed_5 = torch.cat([k_embed_4, k_pass_2], dim=-1)
        k_embed_4 = k_pass_2 = None
        attention_mask_3 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_10 = torch._C._nn.scaled_dot_product_attention(
            q_embed_5,
            k_embed_5,
            value_states_2,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_5 = k_embed_5 = value_states_2 = attention_mask_3 = None
        transpose_6 = attn_output_10.transpose(1, 2)
        attn_output_10 = None
        attn_output_11 = transpose_6.contiguous()
        transpose_6 = None
        reshape_2 = attn_output_11.reshape(1, 45, -1)
        attn_output_11 = None
        attn_output_12 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_13 = torch._C._nn.linear(
            attn_output_12,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_12 = l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_14 = torch.nn.functional.dropout(attn_output_13, 0.0, False, False)
        attn_output_13 = None
        layer_norm_5 = torch.nn.functional.layer_norm(
            hidden_states_8,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_9 = torch._C._nn.linear(
            layer_norm_5,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_5 = l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_10 = torch._C._nn.gelu(hidden_states_9)
        hidden_states_9 = None
        hidden_states_11 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_10 = l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_2 = torch.nn.functional.dropout(hidden_states_11, 0.0, False, False)
        hidden_states_11 = None
        add_10 = mlp_output_2 + attn_output_14
        mlp_output_2 = attn_output_14 = None
        hidden_states_12 = add_10 + hidden_states_8
        add_10 = hidden_states_8 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            hidden_states_12,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_6 = l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_3 = linear_12.view((1, 45, -1, 384))
        linear_12 = None
        qkv_3 = view_3.transpose(1, 2)
        view_3 = None
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
        mul_14 = q_rot_3 * cos_6
        x1_6 = q_rot_3[(Ellipsis, slice(None, 32, None))]
        x2_6 = q_rot_3[(Ellipsis, slice(32, None, None))]
        q_rot_3 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_13 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_15 = cat_13 * sin_6
        cat_13 = None
        q_embed_6 = mul_14 + mul_15
        mul_14 = mul_15 = None
        mul_16 = k_rot_3 * cos_6
        cos_6 = None
        x1_7 = k_rot_3[(Ellipsis, slice(None, 32, None))]
        x2_7 = k_rot_3[(Ellipsis, slice(32, None, None))]
        k_rot_3 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_14 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_17 = cat_14 * sin_6
        cat_14 = sin_6 = None
        k_embed_6 = mul_16 + mul_17
        mul_16 = mul_17 = None
        q_embed_7 = torch.cat([q_embed_6, q_pass_3], dim=-1)
        q_embed_6 = q_pass_3 = None
        k_embed_7 = torch.cat([k_embed_6, k_pass_3], dim=-1)
        k_embed_6 = k_pass_3 = None
        attention_mask_4 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_15 = torch._C._nn.scaled_dot_product_attention(
            q_embed_7,
            k_embed_7,
            value_states_3,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_7 = k_embed_7 = value_states_3 = attention_mask_4 = None
        transpose_8 = attn_output_15.transpose(1, 2)
        attn_output_15 = None
        attn_output_16 = transpose_8.contiguous()
        transpose_8 = None
        reshape_3 = attn_output_16.reshape(1, 45, -1)
        attn_output_16 = None
        attn_output_17 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_18 = torch._C._nn.linear(
            attn_output_17,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_17 = l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_19 = torch.nn.functional.dropout(attn_output_18, 0.0, False, False)
        attn_output_18 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            hidden_states_12,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_13 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_7 = l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_14 = torch._C._nn.gelu(hidden_states_13)
        hidden_states_13 = None
        hidden_states_15 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_14 = l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_3 = torch.nn.functional.dropout(hidden_states_15, 0.0, False, False)
        hidden_states_15 = None
        add_14 = mlp_output_3 + attn_output_19
        mlp_output_3 = attn_output_19 = None
        hidden_states_16 = add_14 + hidden_states_12
        add_14 = hidden_states_12 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            hidden_states_16,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_bias_ = (None)
        linear_16 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_8 = l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_4 = linear_16.view((1, 45, -1, 384))
        linear_16 = None
        qkv_4 = view_4.transpose(1, 2)
        view_4 = None
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
        mul_18 = q_rot_4 * cos_7
        x1_8 = q_rot_4[(Ellipsis, slice(None, 32, None))]
        x2_8 = q_rot_4[(Ellipsis, slice(32, None, None))]
        q_rot_4 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_17 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_19 = cat_17 * sin_7
        cat_17 = None
        q_embed_8 = mul_18 + mul_19
        mul_18 = mul_19 = None
        mul_20 = k_rot_4 * cos_7
        cos_7 = None
        x1_9 = k_rot_4[(Ellipsis, slice(None, 32, None))]
        x2_9 = k_rot_4[(Ellipsis, slice(32, None, None))]
        k_rot_4 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_18 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_21 = cat_18 * sin_7
        cat_18 = sin_7 = None
        k_embed_8 = mul_20 + mul_21
        mul_20 = mul_21 = None
        q_embed_9 = torch.cat([q_embed_8, q_pass_4], dim=-1)
        q_embed_8 = q_pass_4 = None
        k_embed_9 = torch.cat([k_embed_8, k_pass_4], dim=-1)
        k_embed_8 = k_pass_4 = None
        attention_mask_5 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            q_embed_9,
            k_embed_9,
            value_states_4,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_9 = k_embed_9 = value_states_4 = attention_mask_5 = None
        transpose_10 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_10.contiguous()
        transpose_10 = None
        reshape_4 = attn_output_21.reshape(1, 45, -1)
        attn_output_21 = None
        attn_output_22 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_22 = l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_24 = torch.nn.functional.dropout(attn_output_23, 0.0, False, False)
        attn_output_23 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            hidden_states_16,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_17 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_9 = l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_18 = torch._C._nn.gelu(hidden_states_17)
        hidden_states_17 = None
        hidden_states_19 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_18 = l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_4 = torch.nn.functional.dropout(hidden_states_19, 0.0, False, False)
        hidden_states_19 = None
        add_18 = mlp_output_4 + attn_output_24
        mlp_output_4 = attn_output_24 = None
        hidden_states_20 = add_18 + hidden_states_16
        add_18 = hidden_states_16 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            hidden_states_20,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_bias_ = (None)
        linear_20 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_10 = l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_5 = linear_20.view((1, 45, -1, 384))
        linear_20 = None
        qkv_5 = view_5.transpose(1, 2)
        view_5 = None
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
        mul_22 = q_rot_5 * cos_8
        x1_10 = q_rot_5[(Ellipsis, slice(None, 32, None))]
        x2_10 = q_rot_5[(Ellipsis, slice(32, None, None))]
        q_rot_5 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_21 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_23 = cat_21 * sin_8
        cat_21 = None
        q_embed_10 = mul_22 + mul_23
        mul_22 = mul_23 = None
        mul_24 = k_rot_5 * cos_8
        cos_8 = None
        x1_11 = k_rot_5[(Ellipsis, slice(None, 32, None))]
        x2_11 = k_rot_5[(Ellipsis, slice(32, None, None))]
        k_rot_5 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_22 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_25 = cat_22 * sin_8
        cat_22 = sin_8 = None
        k_embed_10 = mul_24 + mul_25
        mul_24 = mul_25 = None
        q_embed_11 = torch.cat([q_embed_10, q_pass_5], dim=-1)
        q_embed_10 = q_pass_5 = None
        k_embed_11 = torch.cat([k_embed_10, k_pass_5], dim=-1)
        k_embed_10 = k_pass_5 = None
        attention_mask_6 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_25 = torch._C._nn.scaled_dot_product_attention(
            q_embed_11,
            k_embed_11,
            value_states_5,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_11 = k_embed_11 = value_states_5 = attention_mask_6 = None
        transpose_12 = attn_output_25.transpose(1, 2)
        attn_output_25 = None
        attn_output_26 = transpose_12.contiguous()
        transpose_12 = None
        reshape_5 = attn_output_26.reshape(1, 45, -1)
        attn_output_26 = None
        attn_output_27 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_28 = torch._C._nn.linear(
            attn_output_27,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_27 = l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_29 = torch.nn.functional.dropout(attn_output_28, 0.0, False, False)
        attn_output_28 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            hidden_states_20,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_21 = torch._C._nn.linear(
            layer_norm_11,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_11 = l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_22 = torch._C._nn.gelu(hidden_states_21)
        hidden_states_21 = None
        hidden_states_23 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_22 = l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_5 = torch.nn.functional.dropout(hidden_states_23, 0.0, False, False)
        hidden_states_23 = None
        add_22 = mlp_output_5 + attn_output_29
        mlp_output_5 = attn_output_29 = None
        hidden_states_24 = add_22 + hidden_states_20
        add_22 = hidden_states_20 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_6_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_6_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_6_modules_input_layernorm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_12 = l_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_6 = linear_24.view((1, 45, -1, 384))
        linear_24 = None
        qkv_6 = view_6.transpose(1, 2)
        view_6 = None
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
        mul_26 = q_rot_6 * cos_9
        x1_12 = q_rot_6[(Ellipsis, slice(None, 32, None))]
        x2_12 = q_rot_6[(Ellipsis, slice(32, None, None))]
        q_rot_6 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_25 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_27 = cat_25 * sin_9
        cat_25 = None
        q_embed_12 = mul_26 + mul_27
        mul_26 = mul_27 = None
        mul_28 = k_rot_6 * cos_9
        cos_9 = None
        x1_13 = k_rot_6[(Ellipsis, slice(None, 32, None))]
        x2_13 = k_rot_6[(Ellipsis, slice(32, None, None))]
        k_rot_6 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_26 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_29 = cat_26 * sin_9
        cat_26 = sin_9 = None
        k_embed_12 = mul_28 + mul_29
        mul_28 = mul_29 = None
        q_embed_13 = torch.cat([q_embed_12, q_pass_6], dim=-1)
        q_embed_12 = q_pass_6 = None
        k_embed_13 = torch.cat([k_embed_12, k_pass_6], dim=-1)
        k_embed_12 = k_pass_6 = None
        attention_mask_7 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_30 = torch._C._nn.scaled_dot_product_attention(
            q_embed_13,
            k_embed_13,
            value_states_6,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_13 = k_embed_13 = value_states_6 = attention_mask_7 = None
        transpose_14 = attn_output_30.transpose(1, 2)
        attn_output_30 = None
        attn_output_31 = transpose_14.contiguous()
        transpose_14 = None
        reshape_6 = attn_output_31.reshape(1, 45, -1)
        attn_output_31 = None
        attn_output_32 = reshape_6.contiguous()
        reshape_6 = None
        attn_output_33 = torch._C._nn.linear(
            attn_output_32,
            l_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_32 = l_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_6_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_34 = torch.nn.functional.dropout(attn_output_33, 0.0, False, False)
        attn_output_33 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_25 = torch._C._nn.linear(
            layer_norm_13,
            l_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_13 = l_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_26 = torch._C._nn.gelu(hidden_states_25)
        hidden_states_25 = None
        hidden_states_27 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_26 = l_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_6 = torch.nn.functional.dropout(hidden_states_27, 0.0, False, False)
        hidden_states_27 = None
        add_26 = mlp_output_6 + attn_output_34
        mlp_output_6 = attn_output_34 = None
        hidden_states_28 = add_26 + hidden_states_24
        add_26 = hidden_states_24 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            hidden_states_28,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_7_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_7_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_7_modules_input_layernorm_parameters_bias_ = (None)
        linear_28 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_14 = l_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_7 = linear_28.view((1, 45, -1, 384))
        linear_28 = None
        qkv_7 = view_7.transpose(1, 2)
        view_7 = None
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
        mul_30 = q_rot_7 * cos_10
        x1_14 = q_rot_7[(Ellipsis, slice(None, 32, None))]
        x2_14 = q_rot_7[(Ellipsis, slice(32, None, None))]
        q_rot_7 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_29 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_31 = cat_29 * sin_10
        cat_29 = None
        q_embed_14 = mul_30 + mul_31
        mul_30 = mul_31 = None
        mul_32 = k_rot_7 * cos_10
        cos_10 = None
        x1_15 = k_rot_7[(Ellipsis, slice(None, 32, None))]
        x2_15 = k_rot_7[(Ellipsis, slice(32, None, None))]
        k_rot_7 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_30 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_33 = cat_30 * sin_10
        cat_30 = sin_10 = None
        k_embed_14 = mul_32 + mul_33
        mul_32 = mul_33 = None
        q_embed_15 = torch.cat([q_embed_14, q_pass_7], dim=-1)
        q_embed_14 = q_pass_7 = None
        k_embed_15 = torch.cat([k_embed_14, k_pass_7], dim=-1)
        k_embed_14 = k_pass_7 = None
        attention_mask_8 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_35 = torch._C._nn.scaled_dot_product_attention(
            q_embed_15,
            k_embed_15,
            value_states_7,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_15 = k_embed_15 = value_states_7 = attention_mask_8 = None
        transpose_16 = attn_output_35.transpose(1, 2)
        attn_output_35 = None
        attn_output_36 = transpose_16.contiguous()
        transpose_16 = None
        reshape_7 = attn_output_36.reshape(1, 45, -1)
        attn_output_36 = None
        attn_output_37 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_38 = torch._C._nn.linear(
            attn_output_37,
            l_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_37 = l_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_7_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_39 = torch.nn.functional.dropout(attn_output_38, 0.0, False, False)
        attn_output_38 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            hidden_states_28,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_29 = torch._C._nn.linear(
            layer_norm_15,
            l_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_15 = l_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_30 = torch._C._nn.gelu(hidden_states_29)
        hidden_states_29 = None
        hidden_states_31 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_30 = l_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_7 = torch.nn.functional.dropout(hidden_states_31, 0.0, False, False)
        hidden_states_31 = None
        add_30 = mlp_output_7 + attn_output_39
        mlp_output_7 = attn_output_39 = None
        hidden_states_32 = add_30 + hidden_states_28
        add_30 = hidden_states_28 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            hidden_states_32,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_8_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_8_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_8_modules_input_layernorm_parameters_bias_ = (None)
        linear_32 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_8 = linear_32.view((1, 45, -1, 384))
        linear_32 = None
        qkv_8 = view_8.transpose(1, 2)
        view_8 = None
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
        mul_34 = q_rot_8 * cos_11
        x1_16 = q_rot_8[(Ellipsis, slice(None, 32, None))]
        x2_16 = q_rot_8[(Ellipsis, slice(32, None, None))]
        q_rot_8 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_33 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_35 = cat_33 * sin_11
        cat_33 = None
        q_embed_16 = mul_34 + mul_35
        mul_34 = mul_35 = None
        mul_36 = k_rot_8 * cos_11
        cos_11 = None
        x1_17 = k_rot_8[(Ellipsis, slice(None, 32, None))]
        x2_17 = k_rot_8[(Ellipsis, slice(32, None, None))]
        k_rot_8 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_34 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_37 = cat_34 * sin_11
        cat_34 = sin_11 = None
        k_embed_16 = mul_36 + mul_37
        mul_36 = mul_37 = None
        q_embed_17 = torch.cat([q_embed_16, q_pass_8], dim=-1)
        q_embed_16 = q_pass_8 = None
        k_embed_17 = torch.cat([k_embed_16, k_pass_8], dim=-1)
        k_embed_16 = k_pass_8 = None
        attention_mask_9 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            q_embed_17,
            k_embed_17,
            value_states_8,
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_17 = k_embed_17 = value_states_8 = attention_mask_9 = None
        transpose_18 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_18.contiguous()
        transpose_18 = None
        reshape_8 = attn_output_41.reshape(1, 45, -1)
        attn_output_41 = None
        attn_output_42 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_42 = l_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_8_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_44 = torch.nn.functional.dropout(attn_output_43, 0.0, False, False)
        attn_output_43 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            hidden_states_32,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_33 = torch._C._nn.linear(
            layer_norm_17,
            l_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_17 = l_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_34 = torch._C._nn.gelu(hidden_states_33)
        hidden_states_33 = None
        hidden_states_35 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_34 = l_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_8 = torch.nn.functional.dropout(hidden_states_35, 0.0, False, False)
        hidden_states_35 = None
        add_34 = mlp_output_8 + attn_output_44
        mlp_output_8 = attn_output_44 = None
        hidden_states_36 = add_34 + hidden_states_32
        add_34 = hidden_states_32 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            hidden_states_36,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_9_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_9_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_9_modules_input_layernorm_parameters_bias_ = (None)
        linear_36 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_18 = l_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_9 = linear_36.view((1, 45, -1, 384))
        linear_36 = None
        qkv_9 = view_9.transpose(1, 2)
        view_9 = None
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
        mul_38 = q_rot_9 * cos_12
        x1_18 = q_rot_9[(Ellipsis, slice(None, 32, None))]
        x2_18 = q_rot_9[(Ellipsis, slice(32, None, None))]
        q_rot_9 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_37 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_39 = cat_37 * sin_12
        cat_37 = None
        q_embed_18 = mul_38 + mul_39
        mul_38 = mul_39 = None
        mul_40 = k_rot_9 * cos_12
        cos_12 = None
        x1_19 = k_rot_9[(Ellipsis, slice(None, 32, None))]
        x2_19 = k_rot_9[(Ellipsis, slice(32, None, None))]
        k_rot_9 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_38 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_41 = cat_38 * sin_12
        cat_38 = sin_12 = None
        k_embed_18 = mul_40 + mul_41
        mul_40 = mul_41 = None
        q_embed_19 = torch.cat([q_embed_18, q_pass_9], dim=-1)
        q_embed_18 = q_pass_9 = None
        k_embed_19 = torch.cat([k_embed_18, k_pass_9], dim=-1)
        k_embed_18 = k_pass_9 = None
        attention_mask_10 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_45 = torch._C._nn.scaled_dot_product_attention(
            q_embed_19,
            k_embed_19,
            value_states_9,
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_19 = k_embed_19 = value_states_9 = attention_mask_10 = None
        transpose_20 = attn_output_45.transpose(1, 2)
        attn_output_45 = None
        attn_output_46 = transpose_20.contiguous()
        transpose_20 = None
        reshape_9 = attn_output_46.reshape(1, 45, -1)
        attn_output_46 = None
        attn_output_47 = reshape_9.contiguous()
        reshape_9 = None
        attn_output_48 = torch._C._nn.linear(
            attn_output_47,
            l_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_47 = l_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_9_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_49 = torch.nn.functional.dropout(attn_output_48, 0.0, False, False)
        attn_output_48 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            hidden_states_36,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_37 = torch._C._nn.linear(
            layer_norm_19,
            l_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_19 = l_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_38 = torch._C._nn.gelu(hidden_states_37)
        hidden_states_37 = None
        hidden_states_39 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_38 = l_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_9 = torch.nn.functional.dropout(hidden_states_39, 0.0, False, False)
        hidden_states_39 = None
        add_38 = mlp_output_9 + attn_output_49
        mlp_output_9 = attn_output_49 = None
        hidden_states_40 = add_38 + hidden_states_36
        add_38 = hidden_states_36 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            hidden_states_40,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_10_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_10_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_10_modules_input_layernorm_parameters_bias_ = (None)
        linear_40 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_20 = l_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_10 = linear_40.view((1, 45, -1, 384))
        linear_40 = None
        qkv_10 = view_10.transpose(1, 2)
        view_10 = None
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
        mul_42 = q_rot_10 * cos_13
        x1_20 = q_rot_10[(Ellipsis, slice(None, 32, None))]
        x2_20 = q_rot_10[(Ellipsis, slice(32, None, None))]
        q_rot_10 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_41 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_43 = cat_41 * sin_13
        cat_41 = None
        q_embed_20 = mul_42 + mul_43
        mul_42 = mul_43 = None
        mul_44 = k_rot_10 * cos_13
        cos_13 = None
        x1_21 = k_rot_10[(Ellipsis, slice(None, 32, None))]
        x2_21 = k_rot_10[(Ellipsis, slice(32, None, None))]
        k_rot_10 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_42 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_45 = cat_42 * sin_13
        cat_42 = sin_13 = None
        k_embed_20 = mul_44 + mul_45
        mul_44 = mul_45 = None
        q_embed_21 = torch.cat([q_embed_20, q_pass_10], dim=-1)
        q_embed_20 = q_pass_10 = None
        k_embed_21 = torch.cat([k_embed_20, k_pass_10], dim=-1)
        k_embed_20 = k_pass_10 = None
        attention_mask_11 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_50 = torch._C._nn.scaled_dot_product_attention(
            q_embed_21,
            k_embed_21,
            value_states_10,
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_21 = k_embed_21 = value_states_10 = attention_mask_11 = None
        transpose_22 = attn_output_50.transpose(1, 2)
        attn_output_50 = None
        attn_output_51 = transpose_22.contiguous()
        transpose_22 = None
        reshape_10 = attn_output_51.reshape(1, 45, -1)
        attn_output_51 = None
        attn_output_52 = reshape_10.contiguous()
        reshape_10 = None
        attn_output_53 = torch._C._nn.linear(
            attn_output_52,
            l_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_52 = l_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_10_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_54 = torch.nn.functional.dropout(attn_output_53, 0.0, False, False)
        attn_output_53 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            hidden_states_40,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_41 = torch._C._nn.linear(
            layer_norm_21,
            l_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_21 = l_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_42 = torch._C._nn.gelu(hidden_states_41)
        hidden_states_41 = None
        hidden_states_43 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_42 = l_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_10 = torch.nn.functional.dropout(hidden_states_43, 0.0, False, False)
        hidden_states_43 = None
        add_42 = mlp_output_10 + attn_output_54
        mlp_output_10 = attn_output_54 = None
        hidden_states_44 = add_42 + hidden_states_40
        add_42 = hidden_states_40 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            hidden_states_44,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_11_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_11_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_11_modules_input_layernorm_parameters_bias_ = (None)
        linear_44 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_22 = l_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_11 = linear_44.view((1, 45, -1, 384))
        linear_44 = None
        qkv_11 = view_11.transpose(1, 2)
        view_11 = None
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
        mul_46 = q_rot_11 * cos_14
        x1_22 = q_rot_11[(Ellipsis, slice(None, 32, None))]
        x2_22 = q_rot_11[(Ellipsis, slice(32, None, None))]
        q_rot_11 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_45 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_47 = cat_45 * sin_14
        cat_45 = None
        q_embed_22 = mul_46 + mul_47
        mul_46 = mul_47 = None
        mul_48 = k_rot_11 * cos_14
        cos_14 = None
        x1_23 = k_rot_11[(Ellipsis, slice(None, 32, None))]
        x2_23 = k_rot_11[(Ellipsis, slice(32, None, None))]
        k_rot_11 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_46 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_49 = cat_46 * sin_14
        cat_46 = sin_14 = None
        k_embed_22 = mul_48 + mul_49
        mul_48 = mul_49 = None
        q_embed_23 = torch.cat([q_embed_22, q_pass_11], dim=-1)
        q_embed_22 = q_pass_11 = None
        k_embed_23 = torch.cat([k_embed_22, k_pass_11], dim=-1)
        k_embed_22 = k_pass_11 = None
        attention_mask_12 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_55 = torch._C._nn.scaled_dot_product_attention(
            q_embed_23,
            k_embed_23,
            value_states_11,
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_23 = k_embed_23 = value_states_11 = attention_mask_12 = None
        transpose_24 = attn_output_55.transpose(1, 2)
        attn_output_55 = None
        attn_output_56 = transpose_24.contiguous()
        transpose_24 = None
        reshape_11 = attn_output_56.reshape(1, 45, -1)
        attn_output_56 = None
        attn_output_57 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_58 = torch._C._nn.linear(
            attn_output_57,
            l_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_57 = l_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_11_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_59 = torch.nn.functional.dropout(attn_output_58, 0.0, False, False)
        attn_output_58 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            hidden_states_44,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_45 = torch._C._nn.linear(
            layer_norm_23,
            l_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_23 = l_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_46 = torch._C._nn.gelu(hidden_states_45)
        hidden_states_45 = None
        hidden_states_47 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_46 = l_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_11 = torch.nn.functional.dropout(hidden_states_47, 0.0, False, False)
        hidden_states_47 = None
        add_46 = mlp_output_11 + attn_output_59
        mlp_output_11 = attn_output_59 = None
        hidden_states_48 = add_46 + hidden_states_44
        add_46 = hidden_states_44 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            hidden_states_48,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_12_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_12_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_12_modules_input_layernorm_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_24 = l_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_12 = linear_48.view((1, 45, -1, 384))
        linear_48 = None
        qkv_12 = view_12.transpose(1, 2)
        view_12 = None
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
        mul_50 = q_rot_12 * cos_15
        x1_24 = q_rot_12[(Ellipsis, slice(None, 32, None))]
        x2_24 = q_rot_12[(Ellipsis, slice(32, None, None))]
        q_rot_12 = None
        neg_24 = -x2_24
        x2_24 = None
        cat_49 = torch.cat((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        mul_51 = cat_49 * sin_15
        cat_49 = None
        q_embed_24 = mul_50 + mul_51
        mul_50 = mul_51 = None
        mul_52 = k_rot_12 * cos_15
        cos_15 = None
        x1_25 = k_rot_12[(Ellipsis, slice(None, 32, None))]
        x2_25 = k_rot_12[(Ellipsis, slice(32, None, None))]
        k_rot_12 = None
        neg_25 = -x2_25
        x2_25 = None
        cat_50 = torch.cat((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        mul_53 = cat_50 * sin_15
        cat_50 = sin_15 = None
        k_embed_24 = mul_52 + mul_53
        mul_52 = mul_53 = None
        q_embed_25 = torch.cat([q_embed_24, q_pass_12], dim=-1)
        q_embed_24 = q_pass_12 = None
        k_embed_25 = torch.cat([k_embed_24, k_pass_12], dim=-1)
        k_embed_24 = k_pass_12 = None
        attention_mask_13 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            q_embed_25,
            k_embed_25,
            value_states_12,
            attn_mask=attention_mask_13,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_25 = k_embed_25 = value_states_12 = attention_mask_13 = None
        transpose_26 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_26.contiguous()
        transpose_26 = None
        reshape_12 = attn_output_61.reshape(1, 45, -1)
        attn_output_61 = None
        attn_output_62 = reshape_12.contiguous()
        reshape_12 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_62 = l_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_12_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_64 = torch.nn.functional.dropout(attn_output_63, 0.0, False, False)
        attn_output_63 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            hidden_states_48,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_49 = torch._C._nn.linear(
            layer_norm_25,
            l_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_25 = l_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_50 = torch._C._nn.gelu(hidden_states_49)
        hidden_states_49 = None
        hidden_states_51 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_50 = l_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_12 = torch.nn.functional.dropout(hidden_states_51, 0.0, False, False)
        hidden_states_51 = None
        add_50 = mlp_output_12 + attn_output_64
        mlp_output_12 = attn_output_64 = None
        hidden_states_52 = add_50 + hidden_states_48
        add_50 = hidden_states_48 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            hidden_states_52,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_13_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_13_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_13_modules_input_layernorm_parameters_bias_ = (None)
        linear_52 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_26 = l_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_13 = linear_52.view((1, 45, -1, 384))
        linear_52 = None
        qkv_13 = view_13.transpose(1, 2)
        view_13 = None
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
        mul_54 = q_rot_13 * cos_16
        x1_26 = q_rot_13[(Ellipsis, slice(None, 32, None))]
        x2_26 = q_rot_13[(Ellipsis, slice(32, None, None))]
        q_rot_13 = None
        neg_26 = -x2_26
        x2_26 = None
        cat_53 = torch.cat((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        mul_55 = cat_53 * sin_16
        cat_53 = None
        q_embed_26 = mul_54 + mul_55
        mul_54 = mul_55 = None
        mul_56 = k_rot_13 * cos_16
        cos_16 = None
        x1_27 = k_rot_13[(Ellipsis, slice(None, 32, None))]
        x2_27 = k_rot_13[(Ellipsis, slice(32, None, None))]
        k_rot_13 = None
        neg_27 = -x2_27
        x2_27 = None
        cat_54 = torch.cat((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        mul_57 = cat_54 * sin_16
        cat_54 = sin_16 = None
        k_embed_26 = mul_56 + mul_57
        mul_56 = mul_57 = None
        q_embed_27 = torch.cat([q_embed_26, q_pass_13], dim=-1)
        q_embed_26 = q_pass_13 = None
        k_embed_27 = torch.cat([k_embed_26, k_pass_13], dim=-1)
        k_embed_26 = k_pass_13 = None
        attention_mask_14 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_65 = torch._C._nn.scaled_dot_product_attention(
            q_embed_27,
            k_embed_27,
            value_states_13,
            attn_mask=attention_mask_14,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_27 = k_embed_27 = value_states_13 = attention_mask_14 = None
        transpose_28 = attn_output_65.transpose(1, 2)
        attn_output_65 = None
        attn_output_66 = transpose_28.contiguous()
        transpose_28 = None
        reshape_13 = attn_output_66.reshape(1, 45, -1)
        attn_output_66 = None
        attn_output_67 = reshape_13.contiguous()
        reshape_13 = None
        attn_output_68 = torch._C._nn.linear(
            attn_output_67,
            l_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_67 = l_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_13_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_69 = torch.nn.functional.dropout(attn_output_68, 0.0, False, False)
        attn_output_68 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            hidden_states_52,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_53 = torch._C._nn.linear(
            layer_norm_27,
            l_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_27 = l_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_54 = torch._C._nn.gelu(hidden_states_53)
        hidden_states_53 = None
        hidden_states_55 = torch._C._nn.linear(
            hidden_states_54,
            l_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_54 = l_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_13 = torch.nn.functional.dropout(hidden_states_55, 0.0, False, False)
        hidden_states_55 = None
        add_54 = mlp_output_13 + attn_output_69
        mlp_output_13 = attn_output_69 = None
        hidden_states_56 = add_54 + hidden_states_52
        add_54 = hidden_states_52 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            hidden_states_56,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_14_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_14_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_14_modules_input_layernorm_parameters_bias_ = (None)
        linear_56 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_28 = l_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_14 = linear_56.view((1, 45, -1, 384))
        linear_56 = None
        qkv_14 = view_14.transpose(1, 2)
        view_14 = None
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
        mul_58 = q_rot_14 * cos_17
        x1_28 = q_rot_14[(Ellipsis, slice(None, 32, None))]
        x2_28 = q_rot_14[(Ellipsis, slice(32, None, None))]
        q_rot_14 = None
        neg_28 = -x2_28
        x2_28 = None
        cat_57 = torch.cat((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        mul_59 = cat_57 * sin_17
        cat_57 = None
        q_embed_28 = mul_58 + mul_59
        mul_58 = mul_59 = None
        mul_60 = k_rot_14 * cos_17
        cos_17 = None
        x1_29 = k_rot_14[(Ellipsis, slice(None, 32, None))]
        x2_29 = k_rot_14[(Ellipsis, slice(32, None, None))]
        k_rot_14 = None
        neg_29 = -x2_29
        x2_29 = None
        cat_58 = torch.cat((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        mul_61 = cat_58 * sin_17
        cat_58 = sin_17 = None
        k_embed_28 = mul_60 + mul_61
        mul_60 = mul_61 = None
        q_embed_29 = torch.cat([q_embed_28, q_pass_14], dim=-1)
        q_embed_28 = q_pass_14 = None
        k_embed_29 = torch.cat([k_embed_28, k_pass_14], dim=-1)
        k_embed_28 = k_pass_14 = None
        attention_mask_15 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_70 = torch._C._nn.scaled_dot_product_attention(
            q_embed_29,
            k_embed_29,
            value_states_14,
            attn_mask=attention_mask_15,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_29 = k_embed_29 = value_states_14 = attention_mask_15 = None
        transpose_30 = attn_output_70.transpose(1, 2)
        attn_output_70 = None
        attn_output_71 = transpose_30.contiguous()
        transpose_30 = None
        reshape_14 = attn_output_71.reshape(1, 45, -1)
        attn_output_71 = None
        attn_output_72 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_73 = torch._C._nn.linear(
            attn_output_72,
            l_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_72 = l_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_14_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_74 = torch.nn.functional.dropout(attn_output_73, 0.0, False, False)
        attn_output_73 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            hidden_states_56,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_57 = torch._C._nn.linear(
            layer_norm_29,
            l_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_29 = l_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_58 = torch._C._nn.gelu(hidden_states_57)
        hidden_states_57 = None
        hidden_states_59 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_58 = l_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_14 = torch.nn.functional.dropout(hidden_states_59, 0.0, False, False)
        hidden_states_59 = None
        add_58 = mlp_output_14 + attn_output_74
        mlp_output_14 = attn_output_74 = None
        hidden_states_60 = add_58 + hidden_states_56
        add_58 = hidden_states_56 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            hidden_states_60,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_15_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_15_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_15_modules_input_layernorm_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_30 = l_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_15 = linear_60.view((1, 45, -1, 384))
        linear_60 = None
        qkv_15 = view_15.transpose(1, 2)
        view_15 = None
        chunk_15 = qkv_15.chunk(3, dim=-1)
        qkv_15 = None
        query_states_15 = chunk_15[0]
        key_states_15 = chunk_15[1]
        value_states_15 = chunk_15[2]
        chunk_15 = None
        cos_18 = cos_2.unsqueeze(1)
        sin_18 = sin_2.unsqueeze(1)
        q_rot_15 = query_states_15[(Ellipsis, slice(None, 64, None))]
        q_pass_15 = query_states_15[(Ellipsis, slice(64, None, None))]
        query_states_15 = None
        k_rot_15 = key_states_15[(Ellipsis, slice(None, 64, None))]
        k_pass_15 = key_states_15[(Ellipsis, slice(64, None, None))]
        key_states_15 = None
        mul_62 = q_rot_15 * cos_18
        x1_30 = q_rot_15[(Ellipsis, slice(None, 32, None))]
        x2_30 = q_rot_15[(Ellipsis, slice(32, None, None))]
        q_rot_15 = None
        neg_30 = -x2_30
        x2_30 = None
        cat_61 = torch.cat((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        mul_63 = cat_61 * sin_18
        cat_61 = None
        q_embed_30 = mul_62 + mul_63
        mul_62 = mul_63 = None
        mul_64 = k_rot_15 * cos_18
        cos_18 = None
        x1_31 = k_rot_15[(Ellipsis, slice(None, 32, None))]
        x2_31 = k_rot_15[(Ellipsis, slice(32, None, None))]
        k_rot_15 = None
        neg_31 = -x2_31
        x2_31 = None
        cat_62 = torch.cat((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        mul_65 = cat_62 * sin_18
        cat_62 = sin_18 = None
        k_embed_30 = mul_64 + mul_65
        mul_64 = mul_65 = None
        q_embed_31 = torch.cat([q_embed_30, q_pass_15], dim=-1)
        q_embed_30 = q_pass_15 = None
        k_embed_31 = torch.cat([k_embed_30, k_pass_15], dim=-1)
        k_embed_30 = k_pass_15 = None
        attention_mask_16 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_75 = torch._C._nn.scaled_dot_product_attention(
            q_embed_31,
            k_embed_31,
            value_states_15,
            attn_mask=attention_mask_16,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_31 = k_embed_31 = value_states_15 = attention_mask_16 = None
        transpose_32 = attn_output_75.transpose(1, 2)
        attn_output_75 = None
        attn_output_76 = transpose_32.contiguous()
        transpose_32 = None
        reshape_15 = attn_output_76.reshape(1, 45, -1)
        attn_output_76 = None
        attn_output_77 = reshape_15.contiguous()
        reshape_15 = None
        attn_output_78 = torch._C._nn.linear(
            attn_output_77,
            l_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_77 = l_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_15_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_79 = torch.nn.functional.dropout(attn_output_78, 0.0, False, False)
        attn_output_78 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            hidden_states_60,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_61 = torch._C._nn.linear(
            layer_norm_31,
            l_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_31 = l_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_62 = torch._C._nn.gelu(hidden_states_61)
        hidden_states_61 = None
        hidden_states_63 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_62 = l_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_15 = torch.nn.functional.dropout(hidden_states_63, 0.0, False, False)
        hidden_states_63 = None
        add_62 = mlp_output_15 + attn_output_79
        mlp_output_15 = attn_output_79 = None
        hidden_states_64 = add_62 + hidden_states_60
        add_62 = hidden_states_60 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            hidden_states_64,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_16_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_16_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_16_modules_input_layernorm_parameters_bias_ = (None)
        linear_64 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_32 = l_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_16 = linear_64.view((1, 45, -1, 384))
        linear_64 = None
        qkv_16 = view_16.transpose(1, 2)
        view_16 = None
        chunk_16 = qkv_16.chunk(3, dim=-1)
        qkv_16 = None
        query_states_16 = chunk_16[0]
        key_states_16 = chunk_16[1]
        value_states_16 = chunk_16[2]
        chunk_16 = None
        cos_19 = cos_2.unsqueeze(1)
        sin_19 = sin_2.unsqueeze(1)
        q_rot_16 = query_states_16[(Ellipsis, slice(None, 64, None))]
        q_pass_16 = query_states_16[(Ellipsis, slice(64, None, None))]
        query_states_16 = None
        k_rot_16 = key_states_16[(Ellipsis, slice(None, 64, None))]
        k_pass_16 = key_states_16[(Ellipsis, slice(64, None, None))]
        key_states_16 = None
        mul_66 = q_rot_16 * cos_19
        x1_32 = q_rot_16[(Ellipsis, slice(None, 32, None))]
        x2_32 = q_rot_16[(Ellipsis, slice(32, None, None))]
        q_rot_16 = None
        neg_32 = -x2_32
        x2_32 = None
        cat_65 = torch.cat((neg_32, x1_32), dim=-1)
        neg_32 = x1_32 = None
        mul_67 = cat_65 * sin_19
        cat_65 = None
        q_embed_32 = mul_66 + mul_67
        mul_66 = mul_67 = None
        mul_68 = k_rot_16 * cos_19
        cos_19 = None
        x1_33 = k_rot_16[(Ellipsis, slice(None, 32, None))]
        x2_33 = k_rot_16[(Ellipsis, slice(32, None, None))]
        k_rot_16 = None
        neg_33 = -x2_33
        x2_33 = None
        cat_66 = torch.cat((neg_33, x1_33), dim=-1)
        neg_33 = x1_33 = None
        mul_69 = cat_66 * sin_19
        cat_66 = sin_19 = None
        k_embed_32 = mul_68 + mul_69
        mul_68 = mul_69 = None
        q_embed_33 = torch.cat([q_embed_32, q_pass_16], dim=-1)
        q_embed_32 = q_pass_16 = None
        k_embed_33 = torch.cat([k_embed_32, k_pass_16], dim=-1)
        k_embed_32 = k_pass_16 = None
        attention_mask_17 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_80 = torch._C._nn.scaled_dot_product_attention(
            q_embed_33,
            k_embed_33,
            value_states_16,
            attn_mask=attention_mask_17,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_33 = k_embed_33 = value_states_16 = attention_mask_17 = None
        transpose_34 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_34.contiguous()
        transpose_34 = None
        reshape_16 = attn_output_81.reshape(1, 45, -1)
        attn_output_81 = None
        attn_output_82 = reshape_16.contiguous()
        reshape_16 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_82 = l_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_16_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_84 = torch.nn.functional.dropout(attn_output_83, 0.0, False, False)
        attn_output_83 = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            hidden_states_64,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_16_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_16_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_65 = torch._C._nn.linear(
            layer_norm_33,
            l_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_33 = l_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_66 = torch._C._nn.gelu(hidden_states_65)
        hidden_states_65 = None
        hidden_states_67 = torch._C._nn.linear(
            hidden_states_66,
            l_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_66 = l_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_16 = torch.nn.functional.dropout(hidden_states_67, 0.0, False, False)
        hidden_states_67 = None
        add_66 = mlp_output_16 + attn_output_84
        mlp_output_16 = attn_output_84 = None
        hidden_states_68 = add_66 + hidden_states_64
        add_66 = hidden_states_64 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            hidden_states_68,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_17_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_17_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_17_modules_input_layernorm_parameters_bias_ = (None)
        linear_68 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_34 = l_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_17 = linear_68.view((1, 45, -1, 384))
        linear_68 = None
        qkv_17 = view_17.transpose(1, 2)
        view_17 = None
        chunk_17 = qkv_17.chunk(3, dim=-1)
        qkv_17 = None
        query_states_17 = chunk_17[0]
        key_states_17 = chunk_17[1]
        value_states_17 = chunk_17[2]
        chunk_17 = None
        cos_20 = cos_2.unsqueeze(1)
        sin_20 = sin_2.unsqueeze(1)
        q_rot_17 = query_states_17[(Ellipsis, slice(None, 64, None))]
        q_pass_17 = query_states_17[(Ellipsis, slice(64, None, None))]
        query_states_17 = None
        k_rot_17 = key_states_17[(Ellipsis, slice(None, 64, None))]
        k_pass_17 = key_states_17[(Ellipsis, slice(64, None, None))]
        key_states_17 = None
        mul_70 = q_rot_17 * cos_20
        x1_34 = q_rot_17[(Ellipsis, slice(None, 32, None))]
        x2_34 = q_rot_17[(Ellipsis, slice(32, None, None))]
        q_rot_17 = None
        neg_34 = -x2_34
        x2_34 = None
        cat_69 = torch.cat((neg_34, x1_34), dim=-1)
        neg_34 = x1_34 = None
        mul_71 = cat_69 * sin_20
        cat_69 = None
        q_embed_34 = mul_70 + mul_71
        mul_70 = mul_71 = None
        mul_72 = k_rot_17 * cos_20
        cos_20 = None
        x1_35 = k_rot_17[(Ellipsis, slice(None, 32, None))]
        x2_35 = k_rot_17[(Ellipsis, slice(32, None, None))]
        k_rot_17 = None
        neg_35 = -x2_35
        x2_35 = None
        cat_70 = torch.cat((neg_35, x1_35), dim=-1)
        neg_35 = x1_35 = None
        mul_73 = cat_70 * sin_20
        cat_70 = sin_20 = None
        k_embed_34 = mul_72 + mul_73
        mul_72 = mul_73 = None
        q_embed_35 = torch.cat([q_embed_34, q_pass_17], dim=-1)
        q_embed_34 = q_pass_17 = None
        k_embed_35 = torch.cat([k_embed_34, k_pass_17], dim=-1)
        k_embed_34 = k_pass_17 = None
        attention_mask_18 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_85 = torch._C._nn.scaled_dot_product_attention(
            q_embed_35,
            k_embed_35,
            value_states_17,
            attn_mask=attention_mask_18,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_35 = k_embed_35 = value_states_17 = attention_mask_18 = None
        transpose_36 = attn_output_85.transpose(1, 2)
        attn_output_85 = None
        attn_output_86 = transpose_36.contiguous()
        transpose_36 = None
        reshape_17 = attn_output_86.reshape(1, 45, -1)
        attn_output_86 = None
        attn_output_87 = reshape_17.contiguous()
        reshape_17 = None
        attn_output_88 = torch._C._nn.linear(
            attn_output_87,
            l_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_87 = l_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_17_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_89 = torch.nn.functional.dropout(attn_output_88, 0.0, False, False)
        attn_output_88 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            hidden_states_68,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_17_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_17_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_69 = torch._C._nn.linear(
            layer_norm_35,
            l_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_35 = l_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_70 = torch._C._nn.gelu(hidden_states_69)
        hidden_states_69 = None
        hidden_states_71 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_70 = l_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_17 = torch.nn.functional.dropout(hidden_states_71, 0.0, False, False)
        hidden_states_71 = None
        add_70 = mlp_output_17 + attn_output_89
        mlp_output_17 = attn_output_89 = None
        hidden_states_72 = add_70 + hidden_states_68
        add_70 = hidden_states_68 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            hidden_states_72,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_18_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_18_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_18_modules_input_layernorm_parameters_bias_ = (None)
        linear_72 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_36 = l_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_18 = linear_72.view((1, 45, -1, 384))
        linear_72 = None
        qkv_18 = view_18.transpose(1, 2)
        view_18 = None
        chunk_18 = qkv_18.chunk(3, dim=-1)
        qkv_18 = None
        query_states_18 = chunk_18[0]
        key_states_18 = chunk_18[1]
        value_states_18 = chunk_18[2]
        chunk_18 = None
        cos_21 = cos_2.unsqueeze(1)
        sin_21 = sin_2.unsqueeze(1)
        q_rot_18 = query_states_18[(Ellipsis, slice(None, 64, None))]
        q_pass_18 = query_states_18[(Ellipsis, slice(64, None, None))]
        query_states_18 = None
        k_rot_18 = key_states_18[(Ellipsis, slice(None, 64, None))]
        k_pass_18 = key_states_18[(Ellipsis, slice(64, None, None))]
        key_states_18 = None
        mul_74 = q_rot_18 * cos_21
        x1_36 = q_rot_18[(Ellipsis, slice(None, 32, None))]
        x2_36 = q_rot_18[(Ellipsis, slice(32, None, None))]
        q_rot_18 = None
        neg_36 = -x2_36
        x2_36 = None
        cat_73 = torch.cat((neg_36, x1_36), dim=-1)
        neg_36 = x1_36 = None
        mul_75 = cat_73 * sin_21
        cat_73 = None
        q_embed_36 = mul_74 + mul_75
        mul_74 = mul_75 = None
        mul_76 = k_rot_18 * cos_21
        cos_21 = None
        x1_37 = k_rot_18[(Ellipsis, slice(None, 32, None))]
        x2_37 = k_rot_18[(Ellipsis, slice(32, None, None))]
        k_rot_18 = None
        neg_37 = -x2_37
        x2_37 = None
        cat_74 = torch.cat((neg_37, x1_37), dim=-1)
        neg_37 = x1_37 = None
        mul_77 = cat_74 * sin_21
        cat_74 = sin_21 = None
        k_embed_36 = mul_76 + mul_77
        mul_76 = mul_77 = None
        q_embed_37 = torch.cat([q_embed_36, q_pass_18], dim=-1)
        q_embed_36 = q_pass_18 = None
        k_embed_37 = torch.cat([k_embed_36, k_pass_18], dim=-1)
        k_embed_36 = k_pass_18 = None
        attention_mask_19 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_90 = torch._C._nn.scaled_dot_product_attention(
            q_embed_37,
            k_embed_37,
            value_states_18,
            attn_mask=attention_mask_19,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_37 = k_embed_37 = value_states_18 = attention_mask_19 = None
        transpose_38 = attn_output_90.transpose(1, 2)
        attn_output_90 = None
        attn_output_91 = transpose_38.contiguous()
        transpose_38 = None
        reshape_18 = attn_output_91.reshape(1, 45, -1)
        attn_output_91 = None
        attn_output_92 = reshape_18.contiguous()
        reshape_18 = None
        attn_output_93 = torch._C._nn.linear(
            attn_output_92,
            l_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_92 = l_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_18_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_94 = torch.nn.functional.dropout(attn_output_93, 0.0, False, False)
        attn_output_93 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            hidden_states_72,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_18_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_18_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_73 = torch._C._nn.linear(
            layer_norm_37,
            l_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_37 = l_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_74 = torch._C._nn.gelu(hidden_states_73)
        hidden_states_73 = None
        hidden_states_75 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_74 = l_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_18 = torch.nn.functional.dropout(hidden_states_75, 0.0, False, False)
        hidden_states_75 = None
        add_74 = mlp_output_18 + attn_output_94
        mlp_output_18 = attn_output_94 = None
        hidden_states_76 = add_74 + hidden_states_72
        add_74 = hidden_states_72 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            hidden_states_76,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_19_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_19_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_19_modules_input_layernorm_parameters_bias_ = (None)
        linear_76 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_38 = l_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_19 = linear_76.view((1, 45, -1, 384))
        linear_76 = None
        qkv_19 = view_19.transpose(1, 2)
        view_19 = None
        chunk_19 = qkv_19.chunk(3, dim=-1)
        qkv_19 = None
        query_states_19 = chunk_19[0]
        key_states_19 = chunk_19[1]
        value_states_19 = chunk_19[2]
        chunk_19 = None
        cos_22 = cos_2.unsqueeze(1)
        sin_22 = sin_2.unsqueeze(1)
        q_rot_19 = query_states_19[(Ellipsis, slice(None, 64, None))]
        q_pass_19 = query_states_19[(Ellipsis, slice(64, None, None))]
        query_states_19 = None
        k_rot_19 = key_states_19[(Ellipsis, slice(None, 64, None))]
        k_pass_19 = key_states_19[(Ellipsis, slice(64, None, None))]
        key_states_19 = None
        mul_78 = q_rot_19 * cos_22
        x1_38 = q_rot_19[(Ellipsis, slice(None, 32, None))]
        x2_38 = q_rot_19[(Ellipsis, slice(32, None, None))]
        q_rot_19 = None
        neg_38 = -x2_38
        x2_38 = None
        cat_77 = torch.cat((neg_38, x1_38), dim=-1)
        neg_38 = x1_38 = None
        mul_79 = cat_77 * sin_22
        cat_77 = None
        q_embed_38 = mul_78 + mul_79
        mul_78 = mul_79 = None
        mul_80 = k_rot_19 * cos_22
        cos_22 = None
        x1_39 = k_rot_19[(Ellipsis, slice(None, 32, None))]
        x2_39 = k_rot_19[(Ellipsis, slice(32, None, None))]
        k_rot_19 = None
        neg_39 = -x2_39
        x2_39 = None
        cat_78 = torch.cat((neg_39, x1_39), dim=-1)
        neg_39 = x1_39 = None
        mul_81 = cat_78 * sin_22
        cat_78 = sin_22 = None
        k_embed_38 = mul_80 + mul_81
        mul_80 = mul_81 = None
        q_embed_39 = torch.cat([q_embed_38, q_pass_19], dim=-1)
        q_embed_38 = q_pass_19 = None
        k_embed_39 = torch.cat([k_embed_38, k_pass_19], dim=-1)
        k_embed_38 = k_pass_19 = None
        attention_mask_20 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_95 = torch._C._nn.scaled_dot_product_attention(
            q_embed_39,
            k_embed_39,
            value_states_19,
            attn_mask=attention_mask_20,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_39 = k_embed_39 = value_states_19 = attention_mask_20 = None
        transpose_40 = attn_output_95.transpose(1, 2)
        attn_output_95 = None
        attn_output_96 = transpose_40.contiguous()
        transpose_40 = None
        reshape_19 = attn_output_96.reshape(1, 45, -1)
        attn_output_96 = None
        attn_output_97 = reshape_19.contiguous()
        reshape_19 = None
        attn_output_98 = torch._C._nn.linear(
            attn_output_97,
            l_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_97 = l_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_19_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_99 = torch.nn.functional.dropout(attn_output_98, 0.0, False, False)
        attn_output_98 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            hidden_states_76,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_19_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_19_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_77 = torch._C._nn.linear(
            layer_norm_39,
            l_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_39 = l_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_78 = torch._C._nn.gelu(hidden_states_77)
        hidden_states_77 = None
        hidden_states_79 = torch._C._nn.linear(
            hidden_states_78,
            l_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_78 = l_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_19 = torch.nn.functional.dropout(hidden_states_79, 0.0, False, False)
        hidden_states_79 = None
        add_78 = mlp_output_19 + attn_output_99
        mlp_output_19 = attn_output_99 = None
        hidden_states_80 = add_78 + hidden_states_76
        add_78 = hidden_states_76 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            hidden_states_80,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_20_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_20_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_20_modules_input_layernorm_parameters_bias_ = (None)
        linear_80 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_40 = l_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_20 = linear_80.view((1, 45, -1, 384))
        linear_80 = None
        qkv_20 = view_20.transpose(1, 2)
        view_20 = None
        chunk_20 = qkv_20.chunk(3, dim=-1)
        qkv_20 = None
        query_states_20 = chunk_20[0]
        key_states_20 = chunk_20[1]
        value_states_20 = chunk_20[2]
        chunk_20 = None
        cos_23 = cos_2.unsqueeze(1)
        sin_23 = sin_2.unsqueeze(1)
        q_rot_20 = query_states_20[(Ellipsis, slice(None, 64, None))]
        q_pass_20 = query_states_20[(Ellipsis, slice(64, None, None))]
        query_states_20 = None
        k_rot_20 = key_states_20[(Ellipsis, slice(None, 64, None))]
        k_pass_20 = key_states_20[(Ellipsis, slice(64, None, None))]
        key_states_20 = None
        mul_82 = q_rot_20 * cos_23
        x1_40 = q_rot_20[(Ellipsis, slice(None, 32, None))]
        x2_40 = q_rot_20[(Ellipsis, slice(32, None, None))]
        q_rot_20 = None
        neg_40 = -x2_40
        x2_40 = None
        cat_81 = torch.cat((neg_40, x1_40), dim=-1)
        neg_40 = x1_40 = None
        mul_83 = cat_81 * sin_23
        cat_81 = None
        q_embed_40 = mul_82 + mul_83
        mul_82 = mul_83 = None
        mul_84 = k_rot_20 * cos_23
        cos_23 = None
        x1_41 = k_rot_20[(Ellipsis, slice(None, 32, None))]
        x2_41 = k_rot_20[(Ellipsis, slice(32, None, None))]
        k_rot_20 = None
        neg_41 = -x2_41
        x2_41 = None
        cat_82 = torch.cat((neg_41, x1_41), dim=-1)
        neg_41 = x1_41 = None
        mul_85 = cat_82 * sin_23
        cat_82 = sin_23 = None
        k_embed_40 = mul_84 + mul_85
        mul_84 = mul_85 = None
        q_embed_41 = torch.cat([q_embed_40, q_pass_20], dim=-1)
        q_embed_40 = q_pass_20 = None
        k_embed_41 = torch.cat([k_embed_40, k_pass_20], dim=-1)
        k_embed_40 = k_pass_20 = None
        attention_mask_21 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_100 = torch._C._nn.scaled_dot_product_attention(
            q_embed_41,
            k_embed_41,
            value_states_20,
            attn_mask=attention_mask_21,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_41 = k_embed_41 = value_states_20 = attention_mask_21 = None
        transpose_42 = attn_output_100.transpose(1, 2)
        attn_output_100 = None
        attn_output_101 = transpose_42.contiguous()
        transpose_42 = None
        reshape_20 = attn_output_101.reshape(1, 45, -1)
        attn_output_101 = None
        attn_output_102 = reshape_20.contiguous()
        reshape_20 = None
        attn_output_103 = torch._C._nn.linear(
            attn_output_102,
            l_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_102 = l_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_20_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_104 = torch.nn.functional.dropout(
            attn_output_103, 0.0, False, False
        )
        attn_output_103 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            hidden_states_80,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_20_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_20_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_81 = torch._C._nn.linear(
            layer_norm_41,
            l_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_41 = l_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_82 = torch._C._nn.gelu(hidden_states_81)
        hidden_states_81 = None
        hidden_states_83 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_82 = l_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_20 = torch.nn.functional.dropout(hidden_states_83, 0.0, False, False)
        hidden_states_83 = None
        add_82 = mlp_output_20 + attn_output_104
        mlp_output_20 = attn_output_104 = None
        hidden_states_84 = add_82 + hidden_states_80
        add_82 = hidden_states_80 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            hidden_states_84,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_21_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_21_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_21_modules_input_layernorm_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_42 = l_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_21 = linear_84.view((1, 45, -1, 384))
        linear_84 = None
        qkv_21 = view_21.transpose(1, 2)
        view_21 = None
        chunk_21 = qkv_21.chunk(3, dim=-1)
        qkv_21 = None
        query_states_21 = chunk_21[0]
        key_states_21 = chunk_21[1]
        value_states_21 = chunk_21[2]
        chunk_21 = None
        cos_24 = cos_2.unsqueeze(1)
        sin_24 = sin_2.unsqueeze(1)
        q_rot_21 = query_states_21[(Ellipsis, slice(None, 64, None))]
        q_pass_21 = query_states_21[(Ellipsis, slice(64, None, None))]
        query_states_21 = None
        k_rot_21 = key_states_21[(Ellipsis, slice(None, 64, None))]
        k_pass_21 = key_states_21[(Ellipsis, slice(64, None, None))]
        key_states_21 = None
        mul_86 = q_rot_21 * cos_24
        x1_42 = q_rot_21[(Ellipsis, slice(None, 32, None))]
        x2_42 = q_rot_21[(Ellipsis, slice(32, None, None))]
        q_rot_21 = None
        neg_42 = -x2_42
        x2_42 = None
        cat_85 = torch.cat((neg_42, x1_42), dim=-1)
        neg_42 = x1_42 = None
        mul_87 = cat_85 * sin_24
        cat_85 = None
        q_embed_42 = mul_86 + mul_87
        mul_86 = mul_87 = None
        mul_88 = k_rot_21 * cos_24
        cos_24 = None
        x1_43 = k_rot_21[(Ellipsis, slice(None, 32, None))]
        x2_43 = k_rot_21[(Ellipsis, slice(32, None, None))]
        k_rot_21 = None
        neg_43 = -x2_43
        x2_43 = None
        cat_86 = torch.cat((neg_43, x1_43), dim=-1)
        neg_43 = x1_43 = None
        mul_89 = cat_86 * sin_24
        cat_86 = sin_24 = None
        k_embed_42 = mul_88 + mul_89
        mul_88 = mul_89 = None
        q_embed_43 = torch.cat([q_embed_42, q_pass_21], dim=-1)
        q_embed_42 = q_pass_21 = None
        k_embed_43 = torch.cat([k_embed_42, k_pass_21], dim=-1)
        k_embed_42 = k_pass_21 = None
        attention_mask_22 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_105 = torch._C._nn.scaled_dot_product_attention(
            q_embed_43,
            k_embed_43,
            value_states_21,
            attn_mask=attention_mask_22,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_43 = k_embed_43 = value_states_21 = attention_mask_22 = None
        transpose_44 = attn_output_105.transpose(1, 2)
        attn_output_105 = None
        attn_output_106 = transpose_44.contiguous()
        transpose_44 = None
        reshape_21 = attn_output_106.reshape(1, 45, -1)
        attn_output_106 = None
        attn_output_107 = reshape_21.contiguous()
        reshape_21 = None
        attn_output_108 = torch._C._nn.linear(
            attn_output_107,
            l_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_107 = l_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_21_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_109 = torch.nn.functional.dropout(
            attn_output_108, 0.0, False, False
        )
        attn_output_108 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            hidden_states_84,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_21_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_21_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_85 = torch._C._nn.linear(
            layer_norm_43,
            l_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_43 = l_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_86 = torch._C._nn.gelu(hidden_states_85)
        hidden_states_85 = None
        hidden_states_87 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_86 = l_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_21 = torch.nn.functional.dropout(hidden_states_87, 0.0, False, False)
        hidden_states_87 = None
        add_86 = mlp_output_21 + attn_output_109
        mlp_output_21 = attn_output_109 = None
        hidden_states_88 = add_86 + hidden_states_84
        add_86 = hidden_states_84 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            hidden_states_88,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_22_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_22_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_22_modules_input_layernorm_parameters_bias_ = (None)
        linear_88 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_44 = l_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_22 = linear_88.view((1, 45, -1, 384))
        linear_88 = None
        qkv_22 = view_22.transpose(1, 2)
        view_22 = None
        chunk_22 = qkv_22.chunk(3, dim=-1)
        qkv_22 = None
        query_states_22 = chunk_22[0]
        key_states_22 = chunk_22[1]
        value_states_22 = chunk_22[2]
        chunk_22 = None
        cos_25 = cos_2.unsqueeze(1)
        sin_25 = sin_2.unsqueeze(1)
        q_rot_22 = query_states_22[(Ellipsis, slice(None, 64, None))]
        q_pass_22 = query_states_22[(Ellipsis, slice(64, None, None))]
        query_states_22 = None
        k_rot_22 = key_states_22[(Ellipsis, slice(None, 64, None))]
        k_pass_22 = key_states_22[(Ellipsis, slice(64, None, None))]
        key_states_22 = None
        mul_90 = q_rot_22 * cos_25
        x1_44 = q_rot_22[(Ellipsis, slice(None, 32, None))]
        x2_44 = q_rot_22[(Ellipsis, slice(32, None, None))]
        q_rot_22 = None
        neg_44 = -x2_44
        x2_44 = None
        cat_89 = torch.cat((neg_44, x1_44), dim=-1)
        neg_44 = x1_44 = None
        mul_91 = cat_89 * sin_25
        cat_89 = None
        q_embed_44 = mul_90 + mul_91
        mul_90 = mul_91 = None
        mul_92 = k_rot_22 * cos_25
        cos_25 = None
        x1_45 = k_rot_22[(Ellipsis, slice(None, 32, None))]
        x2_45 = k_rot_22[(Ellipsis, slice(32, None, None))]
        k_rot_22 = None
        neg_45 = -x2_45
        x2_45 = None
        cat_90 = torch.cat((neg_45, x1_45), dim=-1)
        neg_45 = x1_45 = None
        mul_93 = cat_90 * sin_25
        cat_90 = sin_25 = None
        k_embed_44 = mul_92 + mul_93
        mul_92 = mul_93 = None
        q_embed_45 = torch.cat([q_embed_44, q_pass_22], dim=-1)
        q_embed_44 = q_pass_22 = None
        k_embed_45 = torch.cat([k_embed_44, k_pass_22], dim=-1)
        k_embed_44 = k_pass_22 = None
        attention_mask_23 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        attn_output_110 = torch._C._nn.scaled_dot_product_attention(
            q_embed_45,
            k_embed_45,
            value_states_22,
            attn_mask=attention_mask_23,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_45 = k_embed_45 = value_states_22 = attention_mask_23 = None
        transpose_46 = attn_output_110.transpose(1, 2)
        attn_output_110 = None
        attn_output_111 = transpose_46.contiguous()
        transpose_46 = None
        reshape_22 = attn_output_111.reshape(1, 45, -1)
        attn_output_111 = None
        attn_output_112 = reshape_22.contiguous()
        reshape_22 = None
        attn_output_113 = torch._C._nn.linear(
            attn_output_112,
            l_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_112 = l_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_22_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_114 = torch.nn.functional.dropout(
            attn_output_113, 0.0, False, False
        )
        attn_output_113 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            hidden_states_88,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_22_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_22_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_89 = torch._C._nn.linear(
            layer_norm_45,
            l_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_45 = l_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_90 = torch._C._nn.gelu(hidden_states_89)
        hidden_states_89 = None
        hidden_states_91 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_90 = l_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_22 = torch.nn.functional.dropout(hidden_states_91, 0.0, False, False)
        hidden_states_91 = None
        add_90 = mlp_output_22 + attn_output_114
        mlp_output_22 = attn_output_114 = None
        hidden_states_92 = add_90 + hidden_states_88
        add_90 = hidden_states_88 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            hidden_states_92,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_23_modules_input_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_23_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_23_modules_input_layernorm_parameters_bias_ = (None)
        linear_92 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_query_key_value_parameters_bias_,
        )
        layer_norm_46 = l_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_query_key_value_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_query_key_value_parameters_bias_ = (None)
        view_23 = linear_92.view((1, 45, -1, 384))
        linear_92 = None
        qkv_23 = view_23.transpose(1, 2)
        view_23 = None
        chunk_23 = qkv_23.chunk(3, dim=-1)
        qkv_23 = None
        query_states_23 = chunk_23[0]
        key_states_23 = chunk_23[1]
        value_states_23 = chunk_23[2]
        chunk_23 = None
        cos_26 = cos_2.unsqueeze(1)
        cos_2 = None
        sin_26 = sin_2.unsqueeze(1)
        sin_2 = None
        q_rot_23 = query_states_23[(Ellipsis, slice(None, 64, None))]
        q_pass_23 = query_states_23[(Ellipsis, slice(64, None, None))]
        query_states_23 = None
        k_rot_23 = key_states_23[(Ellipsis, slice(None, 64, None))]
        k_pass_23 = key_states_23[(Ellipsis, slice(64, None, None))]
        key_states_23 = None
        mul_94 = q_rot_23 * cos_26
        x1_46 = q_rot_23[(Ellipsis, slice(None, 32, None))]
        x2_46 = q_rot_23[(Ellipsis, slice(32, None, None))]
        q_rot_23 = None
        neg_46 = -x2_46
        x2_46 = None
        cat_93 = torch.cat((neg_46, x1_46), dim=-1)
        neg_46 = x1_46 = None
        mul_95 = cat_93 * sin_26
        cat_93 = None
        q_embed_46 = mul_94 + mul_95
        mul_94 = mul_95 = None
        mul_96 = k_rot_23 * cos_26
        cos_26 = None
        x1_47 = k_rot_23[(Ellipsis, slice(None, 32, None))]
        x2_47 = k_rot_23[(Ellipsis, slice(32, None, None))]
        k_rot_23 = None
        neg_47 = -x2_47
        x2_47 = None
        cat_94 = torch.cat((neg_47, x1_47), dim=-1)
        neg_47 = x1_47 = None
        mul_97 = cat_94 * sin_26
        cat_94 = sin_26 = None
        k_embed_46 = mul_96 + mul_97
        mul_96 = mul_97 = None
        q_embed_47 = torch.cat([q_embed_46, q_pass_23], dim=-1)
        q_embed_46 = q_pass_23 = None
        k_embed_47 = torch.cat([k_embed_46, k_pass_23], dim=-1)
        k_embed_46 = k_pass_23 = None
        attention_mask_24 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 45, None),
            )
        ]
        causal_mask = None
        attn_output_115 = torch._C._nn.scaled_dot_product_attention(
            q_embed_47,
            k_embed_47,
            value_states_23,
            attn_mask=attention_mask_24,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_47 = k_embed_47 = value_states_23 = attention_mask_24 = None
        transpose_48 = attn_output_115.transpose(1, 2)
        attn_output_115 = None
        attn_output_116 = transpose_48.contiguous()
        transpose_48 = None
        reshape_23 = attn_output_116.reshape(1, 45, -1)
        attn_output_116 = None
        attn_output_117 = reshape_23.contiguous()
        reshape_23 = None
        attn_output_118 = torch._C._nn.linear(
            attn_output_117,
            l_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_dense_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_dense_parameters_bias_,
        )
        attn_output_117 = l_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_dense_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_23_modules_attention_modules_dense_parameters_bias_ = (None)
        attn_output_119 = torch.nn.functional.dropout(
            attn_output_118, 0.0, False, False
        )
        attn_output_118 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            hidden_states_92,
            (2048,),
            l_self_modules_gpt_neox_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_23_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_gpt_neox_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_23_modules_post_attention_layernorm_parameters_bias_ = (None)
        hidden_states_93 = torch._C._nn.linear(
            layer_norm_47,
            l_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layer_norm_47 = l_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        hidden_states_94 = torch._C._nn.gelu(hidden_states_93)
        hidden_states_93 = None
        hidden_states_95 = torch._C._nn.linear(
            hidden_states_94,
            l_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_94 = l_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_gpt_neox_modules_layers_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        mlp_output_23 = torch.nn.functional.dropout(hidden_states_95, 0.0, False, False)
        hidden_states_95 = None
        add_94 = mlp_output_23 + attn_output_119
        mlp_output_23 = attn_output_119 = None
        hidden_states_96 = add_94 + hidden_states_92
        add_94 = hidden_states_92 = None
        hidden_states_97 = torch.nn.functional.layer_norm(
            hidden_states_96,
            (2048,),
            l_self_modules_gpt_neox_modules_final_layer_norm_parameters_weight_,
            l_self_modules_gpt_neox_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_96 = (
            l_self_modules_gpt_neox_modules_final_layer_norm_parameters_weight_
        ) = l_self_modules_gpt_neox_modules_final_layer_norm_parameters_bias_ = None
        getitem_290 = hidden_states_97[
            (slice(None, None, None), slice(0, None, None), slice(None, None, None))
        ]
        hidden_states_97 = None
        logits = torch._C._nn.linear(
            getitem_290, l_self_modules_embed_out_parameters_weight_, None
        )
        getitem_290 = l_self_modules_embed_out_parameters_weight_ = None
        return (logits,)
