import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_self_modules_word_embeddings_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_word_embeddings_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_h_modules_0_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_12_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_12_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_12_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_12_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_12_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_12_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_12_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_12_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_13_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_13_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_13_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_13_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_13_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_13_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_13_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_13_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_14_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_14_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_14_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_14_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_14_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_14_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_14_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_14_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_15_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_15_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_15_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_15_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_15_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_15_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_15_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_15_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_16_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_16_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_16_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_16_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_16_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_16_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_16_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_16_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_17_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_17_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_17_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_17_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_17_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_17_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_17_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_17_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_18_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_18_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_18_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_18_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_18_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_18_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_18_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_18_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_19_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_19_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_19_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_19_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_19_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_19_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_19_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_19_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_20_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_20_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_20_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_20_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_20_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_20_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_20_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_20_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_21_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_21_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_21_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_21_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_21_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_21_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_21_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_21_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_22_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_22_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_22_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_22_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_22_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_22_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_22_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_22_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_23_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_23_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_23_modules_self_attention_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_23_modules_self_attention_modules_query_key_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_23_modules_self_attention_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_23_modules_self_attention_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_23_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_23_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_h_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_ln_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_ln_f_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_embeds_ = L_inputs_embeds_
        l_self_modules_word_embeddings_layernorm_parameters_weight_ = (
            L_self_modules_word_embeddings_layernorm_parameters_weight_
        )
        l_self_modules_word_embeddings_layernorm_parameters_bias_ = (
            L_self_modules_word_embeddings_layernorm_parameters_bias_
        )
        l_attention_mask_ = L_attention_mask_
        l_self_modules_h_modules_0_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_0_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_0_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_1_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_1_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_1_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_2_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_2_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_2_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_3_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_3_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_3_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_4_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_4_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_4_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_5_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_5_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_5_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_5_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_5_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_5_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_5_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_5_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_5_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_5_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_5_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_5_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_5_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_5_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_5_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_5_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_6_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_6_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_6_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_6_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_6_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_6_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_6_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_6_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_6_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_6_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_6_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_6_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_6_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_6_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_6_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_6_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_7_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_7_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_7_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_7_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_7_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_7_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_7_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_7_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_7_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_7_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_7_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_7_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_7_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_7_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_7_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_7_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_8_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_8_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_8_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_8_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_8_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_8_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_8_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_8_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_8_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_8_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_8_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_8_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_8_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_8_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_8_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_8_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_9_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_9_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_9_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_9_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_9_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_9_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_9_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_9_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_9_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_9_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_9_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_9_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_9_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_9_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_9_modules_post_attention_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_9_modules_post_attention_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_10_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_10_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_10_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_10_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_10_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_10_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_10_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_10_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_10_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_10_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_10_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_10_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_10_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_10_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_10_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_h_modules_10_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_h_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_11_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_11_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_11_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_11_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_11_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_11_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_11_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_11_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_11_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_11_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_11_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_11_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_11_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_11_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_11_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_h_modules_11_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_h_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_12_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_12_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_12_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_12_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_12_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_12_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_12_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_12_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_12_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_12_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_12_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_12_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_12_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_12_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_12_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_h_modules_12_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_h_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_13_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_13_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_13_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_13_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_13_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_13_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_13_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_13_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_13_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_13_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_13_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_13_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_13_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_13_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_13_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_h_modules_13_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_h_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_14_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_14_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_14_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_14_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_14_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_14_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_14_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_14_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_14_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_14_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_14_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_14_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_14_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_14_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_14_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_h_modules_14_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_h_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_15_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_15_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_15_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_15_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_15_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_15_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_15_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_15_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_15_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_15_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_15_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_15_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_15_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_15_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_15_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_h_modules_15_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_h_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_16_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_16_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_16_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_16_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_16_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_16_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_16_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_16_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_16_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_16_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_16_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_16_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_16_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_16_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_16_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_h_modules_16_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_h_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_17_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_17_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_17_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_17_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_17_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_17_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_17_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_17_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_17_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_17_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_17_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_17_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_17_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_17_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_17_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_h_modules_17_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_h_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_18_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_18_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_18_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_18_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_18_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_18_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_18_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_18_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_18_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_18_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_18_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_18_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_18_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_18_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_18_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_h_modules_18_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_h_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_19_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_19_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_19_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_19_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_19_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_19_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_19_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_19_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_19_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_19_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_19_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_19_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_19_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_19_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_19_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_h_modules_19_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_h_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_20_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_20_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_20_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_20_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_20_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_20_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_20_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_20_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_20_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_20_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_20_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_20_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_20_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_20_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_20_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_h_modules_20_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_h_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_21_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_21_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_21_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_21_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_21_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_21_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_21_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_21_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_21_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_21_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_21_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_21_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_21_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_21_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_21_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_h_modules_21_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_h_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_22_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_22_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_22_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_22_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_22_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_22_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_22_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_22_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_22_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_22_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_22_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_22_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_22_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_22_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_22_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_h_modules_22_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_h_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_h_modules_23_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_h_modules_23_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_h_modules_23_modules_input_layernorm_parameters_bias_ = (
            L_self_modules_h_modules_23_modules_input_layernorm_parameters_bias_
        )
        l_self_modules_h_modules_23_modules_self_attention_modules_query_key_value_parameters_weight_ = L_self_modules_h_modules_23_modules_self_attention_modules_query_key_value_parameters_weight_
        l_self_modules_h_modules_23_modules_self_attention_modules_query_key_value_parameters_bias_ = L_self_modules_h_modules_23_modules_self_attention_modules_query_key_value_parameters_bias_
        l_self_modules_h_modules_23_modules_self_attention_modules_dense_parameters_weight_ = L_self_modules_h_modules_23_modules_self_attention_modules_dense_parameters_weight_
        l_self_modules_h_modules_23_modules_self_attention_modules_dense_parameters_bias_ = L_self_modules_h_modules_23_modules_self_attention_modules_dense_parameters_bias_
        l_self_modules_h_modules_23_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_h_modules_23_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_h_modules_23_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_h_modules_23_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_h_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_h_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_weight_
        l_self_modules_h_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = L_self_modules_h_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_bias_
        l_self_modules_h_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = L_self_modules_h_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_weight_
        l_self_modules_h_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = L_self_modules_h_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_bias_
        l_self_modules_ln_f_parameters_weight_ = L_self_modules_ln_f_parameters_weight_
        l_self_modules_ln_f_parameters_bias_ = L_self_modules_ln_f_parameters_bias_
        cache_position = torch.arange(0, 9, device=device(type="cuda", index=0))
        hidden_states = torch.nn.functional.layer_norm(
            l_inputs_embeds_,
            (1024,),
            l_self_modules_word_embeddings_layernorm_parameters_weight_,
            l_self_modules_word_embeddings_layernorm_parameters_bias_,
            1e-05,
        )
        l_inputs_embeds_ = (
            l_self_modules_word_embeddings_layernorm_parameters_weight_
        ) = l_self_modules_word_embeddings_layernorm_parameters_bias_ = None
        attention_mask = l_attention_mask_.to(device(type="cuda", index=0))
        l_attention_mask_ = None
        base = torch.tensor(
            0.7071067811865476, device=device(type="cuda", index=0), dtype=torch.float32
        )
        powers = torch.arange(
            1, 17, device=device(type="cuda", index=0), dtype=torch.int32
        )
        slopes = torch.pow(base, powers)
        base = powers = None
        cumsum = attention_mask.cumsum(dim=-1)
        sub = cumsum - 1
        cumsum = None
        mul = sub * attention_mask
        sub = None
        arange_tensor = mul[(slice(None, None, None), None, slice(None, None, None))]
        mul = None
        getitem_1 = slopes[(Ellipsis, None)]
        slopes = None
        alibi = getitem_1 * arange_tensor
        getitem_1 = arange_tensor = None
        reshape = alibi.reshape(16, 1, 9)
        alibi = None
        alibi_1 = reshape.to(torch.bfloat16)
        reshape = None
        causal_mask = torch.full(
            (9, 9),
            fill_value=-3.3895313892515355e38,
            dtype=torch.bfloat16,
            device=device(type="cuda", index=0),
        )
        causal_mask_1 = torch.triu(causal_mask, diagonal=1)
        causal_mask = None
        arange_2 = torch.arange(9, device=device(type="cuda", index=0))
        reshape_1 = cache_position.reshape(-1, 1)
        cache_position = None
        gt = arange_2 > reshape_1
        arange_2 = reshape_1 = None
        causal_mask_1 *= gt
        causal_mask_2 = causal_mask_1
        causal_mask_1 = gt = None
        getitem_2 = causal_mask_2[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        causal_mask_2 = None
        causal_mask_3 = getitem_2.expand(1, 1, -1, -1)
        getitem_2 = None
        causal_mask_4 = causal_mask_3.clone()
        causal_mask_3 = None
        getitem_3 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        getitem_4 = attention_mask[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        attention_mask = None
        to_2 = getitem_4.to(device(type="cuda", index=0))
        getitem_4 = None
        padding_mask = getitem_3 + to_2
        getitem_3 = to_2 = None
        padding_mask_1 = padding_mask.__eq__(0)
        padding_mask = None
        getitem_5 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        masked_fill = getitem_5.masked_fill(padding_mask_1, -3.3895313892515355e38)
        getitem_5 = padding_mask_1 = None
        causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ] = masked_fill
        setitem = causal_mask_4
        masked_fill = setitem = None
        layernorm_output = torch.nn.functional.layer_norm(
            hidden_states,
            (1024,),
            l_self_modules_h_modules_0_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_0_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_0_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_0_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv = torch._C._nn.linear(
            layernorm_output,
            l_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output = l_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_1 = fused_qkv.view(1, 9, 16, 3, 64)
        fused_qkv = None
        getitem_6 = fused_qkv_1[(Ellipsis, 0, slice(None, None, None))]
        query_layer = getitem_6.transpose(1, 2)
        getitem_6 = None
        getitem_7 = fused_qkv_1[(Ellipsis, 1, slice(None, None, None))]
        key_layer = getitem_7.transpose(1, 2)
        getitem_7 = None
        getitem_8 = fused_qkv_1[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_1 = None
        value_layer = getitem_8.transpose(1, 2)
        getitem_8 = None
        query_layer_1 = query_layer.reshape(16, -1, 64)
        query_layer = None
        reshape_3 = key_layer.reshape(16, -1, 64)
        key_layer_1 = reshape_3.transpose(-1, -2)
        reshape_3 = None
        value_layer_1 = value_layer.reshape(16, -1, 64)
        attention_scores = alibi_1.baddbmm(
            batch1=query_layer_1, batch2=key_layer_1, beta=1.0, alpha=0.125
        )
        query_layer_1 = key_layer_1 = None
        attn_weights = attention_scores.view(1, 16, 9, -1)
        attention_scores = None
        causal_mask_5 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_1 = attn_weights + causal_mask_5
        attn_weights = causal_mask_5 = None
        softmax = torch.nn.functional.softmax(
            attn_weights_1, dim=-1, dtype=torch.float32
        )
        attn_weights_1 = None
        attention_probs = softmax.to(torch.bfloat16)
        softmax = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.0, False, False
        )
        attention_probs = None
        attention_probs_reshaped = attention_probs_1.view(16, 9, -1)
        attention_probs_1 = None
        context_layer = torch.bmm(attention_probs_reshaped, value_layer_1)
        attention_probs_reshaped = value_layer_1 = None
        x = context_layer.view(1, 16, 9, 64)
        context_layer = None
        x_1 = x.permute(0, 2, 1, 3)
        x = None
        context_layer_1 = x_1.reshape(1, 9, 1024)
        x_1 = None
        output_tensor = torch._C._nn.linear(
            context_layer_1,
            l_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_1 = l_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out = torch.nn.functional.dropout(output_tensor, p=0.0, training=False)
        output_tensor = None
        out_1 = hidden_states + out
        hidden_states = out = None
        layernorm_output_1 = torch.nn.functional.layer_norm(
            out_1,
            (1024,),
            l_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_2 = torch._C._nn.linear(
            layernorm_output_1,
            l_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_1 = l_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_2 = linear_2 * 0.5
        mul_3 = 0.79788456 * linear_2
        mul_4 = 0.044715 * linear_2
        mul_5 = mul_4 * linear_2
        mul_4 = linear_2 = None
        add_3 = 1 + mul_5
        mul_5 = None
        mul_6 = mul_3 * add_3
        mul_3 = add_3 = None
        tanh = torch.tanh(mul_6)
        mul_6 = None
        add_4 = 1.0 + tanh
        tanh = None
        hidden_states_1 = mul_2 * add_4
        mul_2 = add_4 = None
        intermediate_output = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_1 = l_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_2 = torch.nn.functional.dropout(intermediate_output, p=0.0, training=False)
        intermediate_output = None
        out_3 = out_1 + out_2
        out_1 = out_2 = None
        layernorm_output_2 = torch.nn.functional.layer_norm(
            out_3,
            (1024,),
            l_self_modules_h_modules_1_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_1_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_1_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_1_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_2 = torch._C._nn.linear(
            layernorm_output_2,
            l_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_2 = l_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_3 = fused_qkv_2.view(1, 9, 16, 3, 64)
        fused_qkv_2 = None
        getitem_10 = fused_qkv_3[(Ellipsis, 0, slice(None, None, None))]
        query_layer_2 = getitem_10.transpose(1, 2)
        getitem_10 = None
        getitem_11 = fused_qkv_3[(Ellipsis, 1, slice(None, None, None))]
        key_layer_2 = getitem_11.transpose(1, 2)
        getitem_11 = None
        getitem_12 = fused_qkv_3[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_3 = None
        value_layer_2 = getitem_12.transpose(1, 2)
        getitem_12 = None
        query_layer_3 = query_layer_2.reshape(16, -1, 64)
        query_layer_2 = None
        reshape_7 = key_layer_2.reshape(16, -1, 64)
        key_layer_3 = reshape_7.transpose(-1, -2)
        reshape_7 = None
        value_layer_3 = value_layer_2.reshape(16, -1, 64)
        attention_scores_1 = alibi_1.baddbmm(
            batch1=query_layer_3, batch2=key_layer_3, beta=1.0, alpha=0.125
        )
        query_layer_3 = key_layer_3 = None
        attn_weights_2 = attention_scores_1.view(1, 16, 9, -1)
        attention_scores_1 = None
        causal_mask_6 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_3 = attn_weights_2 + causal_mask_6
        attn_weights_2 = causal_mask_6 = None
        softmax_1 = torch.nn.functional.softmax(
            attn_weights_3, dim=-1, dtype=torch.float32
        )
        attn_weights_3 = None
        attention_probs_2 = softmax_1.to(torch.bfloat16)
        softmax_1 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.0, False, False
        )
        attention_probs_2 = None
        attention_probs_reshaped_1 = attention_probs_3.view(16, 9, -1)
        attention_probs_3 = None
        context_layer_2 = torch.bmm(attention_probs_reshaped_1, value_layer_3)
        attention_probs_reshaped_1 = value_layer_3 = None
        x_2 = context_layer_2.view(1, 16, 9, 64)
        context_layer_2 = None
        x_3 = x_2.permute(0, 2, 1, 3)
        x_2 = None
        context_layer_3 = x_3.reshape(1, 9, 1024)
        x_3 = None
        output_tensor_1 = torch._C._nn.linear(
            context_layer_3,
            l_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_3 = l_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_4 = torch.nn.functional.dropout(output_tensor_1, p=0.0, training=False)
        output_tensor_1 = None
        out_5 = out_3 + out_4
        out_3 = out_4 = None
        layernorm_output_3 = torch.nn.functional.layer_norm(
            out_5,
            (1024,),
            l_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_6 = torch._C._nn.linear(
            layernorm_output_3,
            l_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_3 = l_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_8 = linear_6 * 0.5
        mul_9 = 0.79788456 * linear_6
        mul_10 = 0.044715 * linear_6
        mul_11 = mul_10 * linear_6
        mul_10 = linear_6 = None
        add_8 = 1 + mul_11
        mul_11 = None
        mul_12 = mul_9 * add_8
        mul_9 = add_8 = None
        tanh_1 = torch.tanh(mul_12)
        mul_12 = None
        add_9 = 1.0 + tanh_1
        tanh_1 = None
        hidden_states_2 = mul_8 * add_9
        mul_8 = add_9 = None
        intermediate_output_1 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_6 = torch.nn.functional.dropout(
            intermediate_output_1, p=0.0, training=False
        )
        intermediate_output_1 = None
        out_7 = out_5 + out_6
        out_5 = out_6 = None
        layernorm_output_4 = torch.nn.functional.layer_norm(
            out_7,
            (1024,),
            l_self_modules_h_modules_2_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_2_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_2_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_2_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_4 = torch._C._nn.linear(
            layernorm_output_4,
            l_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_4 = l_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_5 = fused_qkv_4.view(1, 9, 16, 3, 64)
        fused_qkv_4 = None
        getitem_14 = fused_qkv_5[(Ellipsis, 0, slice(None, None, None))]
        query_layer_4 = getitem_14.transpose(1, 2)
        getitem_14 = None
        getitem_15 = fused_qkv_5[(Ellipsis, 1, slice(None, None, None))]
        key_layer_4 = getitem_15.transpose(1, 2)
        getitem_15 = None
        getitem_16 = fused_qkv_5[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_5 = None
        value_layer_4 = getitem_16.transpose(1, 2)
        getitem_16 = None
        query_layer_5 = query_layer_4.reshape(16, -1, 64)
        query_layer_4 = None
        reshape_11 = key_layer_4.reshape(16, -1, 64)
        key_layer_5 = reshape_11.transpose(-1, -2)
        reshape_11 = None
        value_layer_5 = value_layer_4.reshape(16, -1, 64)
        attention_scores_2 = alibi_1.baddbmm(
            batch1=query_layer_5, batch2=key_layer_5, beta=1.0, alpha=0.125
        )
        query_layer_5 = key_layer_5 = None
        attn_weights_4 = attention_scores_2.view(1, 16, 9, -1)
        attention_scores_2 = None
        causal_mask_7 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_5 = attn_weights_4 + causal_mask_7
        attn_weights_4 = causal_mask_7 = None
        softmax_2 = torch.nn.functional.softmax(
            attn_weights_5, dim=-1, dtype=torch.float32
        )
        attn_weights_5 = None
        attention_probs_4 = softmax_2.to(torch.bfloat16)
        softmax_2 = None
        attention_probs_5 = torch.nn.functional.dropout(
            attention_probs_4, 0.0, False, False
        )
        attention_probs_4 = None
        attention_probs_reshaped_2 = attention_probs_5.view(16, 9, -1)
        attention_probs_5 = None
        context_layer_4 = torch.bmm(attention_probs_reshaped_2, value_layer_5)
        attention_probs_reshaped_2 = value_layer_5 = None
        x_4 = context_layer_4.view(1, 16, 9, 64)
        context_layer_4 = None
        x_5 = x_4.permute(0, 2, 1, 3)
        x_4 = None
        context_layer_5 = x_5.reshape(1, 9, 1024)
        x_5 = None
        output_tensor_2 = torch._C._nn.linear(
            context_layer_5,
            l_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_5 = l_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_8 = torch.nn.functional.dropout(output_tensor_2, p=0.0, training=False)
        output_tensor_2 = None
        out_9 = out_7 + out_8
        out_7 = out_8 = None
        layernorm_output_5 = torch.nn.functional.layer_norm(
            out_9,
            (1024,),
            l_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_10 = torch._C._nn.linear(
            layernorm_output_5,
            l_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_5 = l_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_14 = linear_10 * 0.5
        mul_15 = 0.79788456 * linear_10
        mul_16 = 0.044715 * linear_10
        mul_17 = mul_16 * linear_10
        mul_16 = linear_10 = None
        add_13 = 1 + mul_17
        mul_17 = None
        mul_18 = mul_15 * add_13
        mul_15 = add_13 = None
        tanh_2 = torch.tanh(mul_18)
        mul_18 = None
        add_14 = 1.0 + tanh_2
        tanh_2 = None
        hidden_states_3 = mul_14 * add_14
        mul_14 = add_14 = None
        intermediate_output_2 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_3 = l_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_10 = torch.nn.functional.dropout(
            intermediate_output_2, p=0.0, training=False
        )
        intermediate_output_2 = None
        out_11 = out_9 + out_10
        out_9 = out_10 = None
        layernorm_output_6 = torch.nn.functional.layer_norm(
            out_11,
            (1024,),
            l_self_modules_h_modules_3_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_3_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_3_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_3_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_6 = torch._C._nn.linear(
            layernorm_output_6,
            l_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_6 = l_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_7 = fused_qkv_6.view(1, 9, 16, 3, 64)
        fused_qkv_6 = None
        getitem_18 = fused_qkv_7[(Ellipsis, 0, slice(None, None, None))]
        query_layer_6 = getitem_18.transpose(1, 2)
        getitem_18 = None
        getitem_19 = fused_qkv_7[(Ellipsis, 1, slice(None, None, None))]
        key_layer_6 = getitem_19.transpose(1, 2)
        getitem_19 = None
        getitem_20 = fused_qkv_7[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_7 = None
        value_layer_6 = getitem_20.transpose(1, 2)
        getitem_20 = None
        query_layer_7 = query_layer_6.reshape(16, -1, 64)
        query_layer_6 = None
        reshape_15 = key_layer_6.reshape(16, -1, 64)
        key_layer_7 = reshape_15.transpose(-1, -2)
        reshape_15 = None
        value_layer_7 = value_layer_6.reshape(16, -1, 64)
        attention_scores_3 = alibi_1.baddbmm(
            batch1=query_layer_7, batch2=key_layer_7, beta=1.0, alpha=0.125
        )
        query_layer_7 = key_layer_7 = None
        attn_weights_6 = attention_scores_3.view(1, 16, 9, -1)
        attention_scores_3 = None
        causal_mask_8 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_7 = attn_weights_6 + causal_mask_8
        attn_weights_6 = causal_mask_8 = None
        softmax_3 = torch.nn.functional.softmax(
            attn_weights_7, dim=-1, dtype=torch.float32
        )
        attn_weights_7 = None
        attention_probs_6 = softmax_3.to(torch.bfloat16)
        softmax_3 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.0, False, False
        )
        attention_probs_6 = None
        attention_probs_reshaped_3 = attention_probs_7.view(16, 9, -1)
        attention_probs_7 = None
        context_layer_6 = torch.bmm(attention_probs_reshaped_3, value_layer_7)
        attention_probs_reshaped_3 = value_layer_7 = None
        x_6 = context_layer_6.view(1, 16, 9, 64)
        context_layer_6 = None
        x_7 = x_6.permute(0, 2, 1, 3)
        x_6 = None
        context_layer_7 = x_7.reshape(1, 9, 1024)
        x_7 = None
        output_tensor_3 = torch._C._nn.linear(
            context_layer_7,
            l_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_7 = l_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_12 = torch.nn.functional.dropout(output_tensor_3, p=0.0, training=False)
        output_tensor_3 = None
        out_13 = out_11 + out_12
        out_11 = out_12 = None
        layernorm_output_7 = torch.nn.functional.layer_norm(
            out_13,
            (1024,),
            l_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_14 = torch._C._nn.linear(
            layernorm_output_7,
            l_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_7 = l_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_20 = linear_14 * 0.5
        mul_21 = 0.79788456 * linear_14
        mul_22 = 0.044715 * linear_14
        mul_23 = mul_22 * linear_14
        mul_22 = linear_14 = None
        add_18 = 1 + mul_23
        mul_23 = None
        mul_24 = mul_21 * add_18
        mul_21 = add_18 = None
        tanh_3 = torch.tanh(mul_24)
        mul_24 = None
        add_19 = 1.0 + tanh_3
        tanh_3 = None
        hidden_states_4 = mul_20 * add_19
        mul_20 = add_19 = None
        intermediate_output_3 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_4 = l_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_14 = torch.nn.functional.dropout(
            intermediate_output_3, p=0.0, training=False
        )
        intermediate_output_3 = None
        out_15 = out_13 + out_14
        out_13 = out_14 = None
        layernorm_output_8 = torch.nn.functional.layer_norm(
            out_15,
            (1024,),
            l_self_modules_h_modules_4_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_4_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_4_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_4_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_8 = torch._C._nn.linear(
            layernorm_output_8,
            l_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_8 = l_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_9 = fused_qkv_8.view(1, 9, 16, 3, 64)
        fused_qkv_8 = None
        getitem_22 = fused_qkv_9[(Ellipsis, 0, slice(None, None, None))]
        query_layer_8 = getitem_22.transpose(1, 2)
        getitem_22 = None
        getitem_23 = fused_qkv_9[(Ellipsis, 1, slice(None, None, None))]
        key_layer_8 = getitem_23.transpose(1, 2)
        getitem_23 = None
        getitem_24 = fused_qkv_9[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_9 = None
        value_layer_8 = getitem_24.transpose(1, 2)
        getitem_24 = None
        query_layer_9 = query_layer_8.reshape(16, -1, 64)
        query_layer_8 = None
        reshape_19 = key_layer_8.reshape(16, -1, 64)
        key_layer_9 = reshape_19.transpose(-1, -2)
        reshape_19 = None
        value_layer_9 = value_layer_8.reshape(16, -1, 64)
        attention_scores_4 = alibi_1.baddbmm(
            batch1=query_layer_9, batch2=key_layer_9, beta=1.0, alpha=0.125
        )
        query_layer_9 = key_layer_9 = None
        attn_weights_8 = attention_scores_4.view(1, 16, 9, -1)
        attention_scores_4 = None
        causal_mask_9 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_9 = attn_weights_8 + causal_mask_9
        attn_weights_8 = causal_mask_9 = None
        softmax_4 = torch.nn.functional.softmax(
            attn_weights_9, dim=-1, dtype=torch.float32
        )
        attn_weights_9 = None
        attention_probs_8 = softmax_4.to(torch.bfloat16)
        softmax_4 = None
        attention_probs_9 = torch.nn.functional.dropout(
            attention_probs_8, 0.0, False, False
        )
        attention_probs_8 = None
        attention_probs_reshaped_4 = attention_probs_9.view(16, 9, -1)
        attention_probs_9 = None
        context_layer_8 = torch.bmm(attention_probs_reshaped_4, value_layer_9)
        attention_probs_reshaped_4 = value_layer_9 = None
        x_8 = context_layer_8.view(1, 16, 9, 64)
        context_layer_8 = None
        x_9 = x_8.permute(0, 2, 1, 3)
        x_8 = None
        context_layer_9 = x_9.reshape(1, 9, 1024)
        x_9 = None
        output_tensor_4 = torch._C._nn.linear(
            context_layer_9,
            l_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_9 = l_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_16 = torch.nn.functional.dropout(output_tensor_4, p=0.0, training=False)
        output_tensor_4 = None
        out_17 = out_15 + out_16
        out_15 = out_16 = None
        layernorm_output_9 = torch.nn.functional.layer_norm(
            out_17,
            (1024,),
            l_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_18 = torch._C._nn.linear(
            layernorm_output_9,
            l_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_9 = l_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_26 = linear_18 * 0.5
        mul_27 = 0.79788456 * linear_18
        mul_28 = 0.044715 * linear_18
        mul_29 = mul_28 * linear_18
        mul_28 = linear_18 = None
        add_23 = 1 + mul_29
        mul_29 = None
        mul_30 = mul_27 * add_23
        mul_27 = add_23 = None
        tanh_4 = torch.tanh(mul_30)
        mul_30 = None
        add_24 = 1.0 + tanh_4
        tanh_4 = None
        hidden_states_5 = mul_26 * add_24
        mul_26 = add_24 = None
        intermediate_output_4 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_5 = l_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_18 = torch.nn.functional.dropout(
            intermediate_output_4, p=0.0, training=False
        )
        intermediate_output_4 = None
        out_19 = out_17 + out_18
        out_17 = out_18 = None
        layernorm_output_10 = torch.nn.functional.layer_norm(
            out_19,
            (1024,),
            l_self_modules_h_modules_5_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_5_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_5_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_5_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_10 = torch._C._nn.linear(
            layernorm_output_10,
            l_self_modules_h_modules_5_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_5_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_10 = l_self_modules_h_modules_5_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_5_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_11 = fused_qkv_10.view(1, 9, 16, 3, 64)
        fused_qkv_10 = None
        getitem_26 = fused_qkv_11[(Ellipsis, 0, slice(None, None, None))]
        query_layer_10 = getitem_26.transpose(1, 2)
        getitem_26 = None
        getitem_27 = fused_qkv_11[(Ellipsis, 1, slice(None, None, None))]
        key_layer_10 = getitem_27.transpose(1, 2)
        getitem_27 = None
        getitem_28 = fused_qkv_11[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_11 = None
        value_layer_10 = getitem_28.transpose(1, 2)
        getitem_28 = None
        query_layer_11 = query_layer_10.reshape(16, -1, 64)
        query_layer_10 = None
        reshape_23 = key_layer_10.reshape(16, -1, 64)
        key_layer_11 = reshape_23.transpose(-1, -2)
        reshape_23 = None
        value_layer_11 = value_layer_10.reshape(16, -1, 64)
        attention_scores_5 = alibi_1.baddbmm(
            batch1=query_layer_11, batch2=key_layer_11, beta=1.0, alpha=0.125
        )
        query_layer_11 = key_layer_11 = None
        attn_weights_10 = attention_scores_5.view(1, 16, 9, -1)
        attention_scores_5 = None
        causal_mask_10 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_11 = attn_weights_10 + causal_mask_10
        attn_weights_10 = causal_mask_10 = None
        softmax_5 = torch.nn.functional.softmax(
            attn_weights_11, dim=-1, dtype=torch.float32
        )
        attn_weights_11 = None
        attention_probs_10 = softmax_5.to(torch.bfloat16)
        softmax_5 = None
        attention_probs_11 = torch.nn.functional.dropout(
            attention_probs_10, 0.0, False, False
        )
        attention_probs_10 = None
        attention_probs_reshaped_5 = attention_probs_11.view(16, 9, -1)
        attention_probs_11 = None
        context_layer_10 = torch.bmm(attention_probs_reshaped_5, value_layer_11)
        attention_probs_reshaped_5 = value_layer_11 = None
        x_10 = context_layer_10.view(1, 16, 9, 64)
        context_layer_10 = None
        x_11 = x_10.permute(0, 2, 1, 3)
        x_10 = None
        context_layer_11 = x_11.reshape(1, 9, 1024)
        x_11 = None
        output_tensor_5 = torch._C._nn.linear(
            context_layer_11,
            l_self_modules_h_modules_5_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_5_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_h_modules_5_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_5_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_20 = torch.nn.functional.dropout(output_tensor_5, p=0.0, training=False)
        output_tensor_5 = None
        out_21 = out_19 + out_20
        out_19 = out_20 = None
        layernorm_output_11 = torch.nn.functional.layer_norm(
            out_21,
            (1024,),
            l_self_modules_h_modules_5_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_5_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_5_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_5_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_22 = torch._C._nn.linear(
            layernorm_output_11,
            l_self_modules_h_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_11 = l_self_modules_h_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_32 = linear_22 * 0.5
        mul_33 = 0.79788456 * linear_22
        mul_34 = 0.044715 * linear_22
        mul_35 = mul_34 * linear_22
        mul_34 = linear_22 = None
        add_28 = 1 + mul_35
        mul_35 = None
        mul_36 = mul_33 * add_28
        mul_33 = add_28 = None
        tanh_5 = torch.tanh(mul_36)
        mul_36 = None
        add_29 = 1.0 + tanh_5
        tanh_5 = None
        hidden_states_6 = mul_32 * add_29
        mul_32 = add_29 = None
        intermediate_output_5 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_h_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_6 = l_self_modules_h_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_22 = torch.nn.functional.dropout(
            intermediate_output_5, p=0.0, training=False
        )
        intermediate_output_5 = None
        out_23 = out_21 + out_22
        out_21 = out_22 = None
        layernorm_output_12 = torch.nn.functional.layer_norm(
            out_23,
            (1024,),
            l_self_modules_h_modules_6_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_6_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_6_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_6_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_12 = torch._C._nn.linear(
            layernorm_output_12,
            l_self_modules_h_modules_6_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_6_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_12 = l_self_modules_h_modules_6_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_6_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_13 = fused_qkv_12.view(1, 9, 16, 3, 64)
        fused_qkv_12 = None
        getitem_30 = fused_qkv_13[(Ellipsis, 0, slice(None, None, None))]
        query_layer_12 = getitem_30.transpose(1, 2)
        getitem_30 = None
        getitem_31 = fused_qkv_13[(Ellipsis, 1, slice(None, None, None))]
        key_layer_12 = getitem_31.transpose(1, 2)
        getitem_31 = None
        getitem_32 = fused_qkv_13[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_13 = None
        value_layer_12 = getitem_32.transpose(1, 2)
        getitem_32 = None
        query_layer_13 = query_layer_12.reshape(16, -1, 64)
        query_layer_12 = None
        reshape_27 = key_layer_12.reshape(16, -1, 64)
        key_layer_13 = reshape_27.transpose(-1, -2)
        reshape_27 = None
        value_layer_13 = value_layer_12.reshape(16, -1, 64)
        attention_scores_6 = alibi_1.baddbmm(
            batch1=query_layer_13, batch2=key_layer_13, beta=1.0, alpha=0.125
        )
        query_layer_13 = key_layer_13 = None
        attn_weights_12 = attention_scores_6.view(1, 16, 9, -1)
        attention_scores_6 = None
        causal_mask_11 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_13 = attn_weights_12 + causal_mask_11
        attn_weights_12 = causal_mask_11 = None
        softmax_6 = torch.nn.functional.softmax(
            attn_weights_13, dim=-1, dtype=torch.float32
        )
        attn_weights_13 = None
        attention_probs_12 = softmax_6.to(torch.bfloat16)
        softmax_6 = None
        attention_probs_13 = torch.nn.functional.dropout(
            attention_probs_12, 0.0, False, False
        )
        attention_probs_12 = None
        attention_probs_reshaped_6 = attention_probs_13.view(16, 9, -1)
        attention_probs_13 = None
        context_layer_12 = torch.bmm(attention_probs_reshaped_6, value_layer_13)
        attention_probs_reshaped_6 = value_layer_13 = None
        x_12 = context_layer_12.view(1, 16, 9, 64)
        context_layer_12 = None
        x_13 = x_12.permute(0, 2, 1, 3)
        x_12 = None
        context_layer_13 = x_13.reshape(1, 9, 1024)
        x_13 = None
        output_tensor_6 = torch._C._nn.linear(
            context_layer_13,
            l_self_modules_h_modules_6_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_6_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_13 = l_self_modules_h_modules_6_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_6_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_24 = torch.nn.functional.dropout(output_tensor_6, p=0.0, training=False)
        output_tensor_6 = None
        out_25 = out_23 + out_24
        out_23 = out_24 = None
        layernorm_output_13 = torch.nn.functional.layer_norm(
            out_25,
            (1024,),
            l_self_modules_h_modules_6_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_6_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_6_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_6_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_26 = torch._C._nn.linear(
            layernorm_output_13,
            l_self_modules_h_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_13 = l_self_modules_h_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_38 = linear_26 * 0.5
        mul_39 = 0.79788456 * linear_26
        mul_40 = 0.044715 * linear_26
        mul_41 = mul_40 * linear_26
        mul_40 = linear_26 = None
        add_33 = 1 + mul_41
        mul_41 = None
        mul_42 = mul_39 * add_33
        mul_39 = add_33 = None
        tanh_6 = torch.tanh(mul_42)
        mul_42 = None
        add_34 = 1.0 + tanh_6
        tanh_6 = None
        hidden_states_7 = mul_38 * add_34
        mul_38 = add_34 = None
        intermediate_output_6 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_h_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_7 = l_self_modules_h_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_26 = torch.nn.functional.dropout(
            intermediate_output_6, p=0.0, training=False
        )
        intermediate_output_6 = None
        out_27 = out_25 + out_26
        out_25 = out_26 = None
        layernorm_output_14 = torch.nn.functional.layer_norm(
            out_27,
            (1024,),
            l_self_modules_h_modules_7_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_7_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_7_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_7_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_14 = torch._C._nn.linear(
            layernorm_output_14,
            l_self_modules_h_modules_7_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_7_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_14 = l_self_modules_h_modules_7_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_7_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_15 = fused_qkv_14.view(1, 9, 16, 3, 64)
        fused_qkv_14 = None
        getitem_34 = fused_qkv_15[(Ellipsis, 0, slice(None, None, None))]
        query_layer_14 = getitem_34.transpose(1, 2)
        getitem_34 = None
        getitem_35 = fused_qkv_15[(Ellipsis, 1, slice(None, None, None))]
        key_layer_14 = getitem_35.transpose(1, 2)
        getitem_35 = None
        getitem_36 = fused_qkv_15[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_15 = None
        value_layer_14 = getitem_36.transpose(1, 2)
        getitem_36 = None
        query_layer_15 = query_layer_14.reshape(16, -1, 64)
        query_layer_14 = None
        reshape_31 = key_layer_14.reshape(16, -1, 64)
        key_layer_15 = reshape_31.transpose(-1, -2)
        reshape_31 = None
        value_layer_15 = value_layer_14.reshape(16, -1, 64)
        attention_scores_7 = alibi_1.baddbmm(
            batch1=query_layer_15, batch2=key_layer_15, beta=1.0, alpha=0.125
        )
        query_layer_15 = key_layer_15 = None
        attn_weights_14 = attention_scores_7.view(1, 16, 9, -1)
        attention_scores_7 = None
        causal_mask_12 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_15 = attn_weights_14 + causal_mask_12
        attn_weights_14 = causal_mask_12 = None
        softmax_7 = torch.nn.functional.softmax(
            attn_weights_15, dim=-1, dtype=torch.float32
        )
        attn_weights_15 = None
        attention_probs_14 = softmax_7.to(torch.bfloat16)
        softmax_7 = None
        attention_probs_15 = torch.nn.functional.dropout(
            attention_probs_14, 0.0, False, False
        )
        attention_probs_14 = None
        attention_probs_reshaped_7 = attention_probs_15.view(16, 9, -1)
        attention_probs_15 = None
        context_layer_14 = torch.bmm(attention_probs_reshaped_7, value_layer_15)
        attention_probs_reshaped_7 = value_layer_15 = None
        x_14 = context_layer_14.view(1, 16, 9, 64)
        context_layer_14 = None
        x_15 = x_14.permute(0, 2, 1, 3)
        x_14 = None
        context_layer_15 = x_15.reshape(1, 9, 1024)
        x_15 = None
        output_tensor_7 = torch._C._nn.linear(
            context_layer_15,
            l_self_modules_h_modules_7_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_7_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_15 = l_self_modules_h_modules_7_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_7_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_28 = torch.nn.functional.dropout(output_tensor_7, p=0.0, training=False)
        output_tensor_7 = None
        out_29 = out_27 + out_28
        out_27 = out_28 = None
        layernorm_output_15 = torch.nn.functional.layer_norm(
            out_29,
            (1024,),
            l_self_modules_h_modules_7_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_7_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_7_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_7_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_30 = torch._C._nn.linear(
            layernorm_output_15,
            l_self_modules_h_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_15 = l_self_modules_h_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_44 = linear_30 * 0.5
        mul_45 = 0.79788456 * linear_30
        mul_46 = 0.044715 * linear_30
        mul_47 = mul_46 * linear_30
        mul_46 = linear_30 = None
        add_38 = 1 + mul_47
        mul_47 = None
        mul_48 = mul_45 * add_38
        mul_45 = add_38 = None
        tanh_7 = torch.tanh(mul_48)
        mul_48 = None
        add_39 = 1.0 + tanh_7
        tanh_7 = None
        hidden_states_8 = mul_44 * add_39
        mul_44 = add_39 = None
        intermediate_output_7 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_h_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_8 = l_self_modules_h_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_30 = torch.nn.functional.dropout(
            intermediate_output_7, p=0.0, training=False
        )
        intermediate_output_7 = None
        out_31 = out_29 + out_30
        out_29 = out_30 = None
        layernorm_output_16 = torch.nn.functional.layer_norm(
            out_31,
            (1024,),
            l_self_modules_h_modules_8_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_8_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_8_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_8_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_16 = torch._C._nn.linear(
            layernorm_output_16,
            l_self_modules_h_modules_8_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_8_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_16 = l_self_modules_h_modules_8_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_8_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_17 = fused_qkv_16.view(1, 9, 16, 3, 64)
        fused_qkv_16 = None
        getitem_38 = fused_qkv_17[(Ellipsis, 0, slice(None, None, None))]
        query_layer_16 = getitem_38.transpose(1, 2)
        getitem_38 = None
        getitem_39 = fused_qkv_17[(Ellipsis, 1, slice(None, None, None))]
        key_layer_16 = getitem_39.transpose(1, 2)
        getitem_39 = None
        getitem_40 = fused_qkv_17[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_17 = None
        value_layer_16 = getitem_40.transpose(1, 2)
        getitem_40 = None
        query_layer_17 = query_layer_16.reshape(16, -1, 64)
        query_layer_16 = None
        reshape_35 = key_layer_16.reshape(16, -1, 64)
        key_layer_17 = reshape_35.transpose(-1, -2)
        reshape_35 = None
        value_layer_17 = value_layer_16.reshape(16, -1, 64)
        attention_scores_8 = alibi_1.baddbmm(
            batch1=query_layer_17, batch2=key_layer_17, beta=1.0, alpha=0.125
        )
        query_layer_17 = key_layer_17 = None
        attn_weights_16 = attention_scores_8.view(1, 16, 9, -1)
        attention_scores_8 = None
        causal_mask_13 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_17 = attn_weights_16 + causal_mask_13
        attn_weights_16 = causal_mask_13 = None
        softmax_8 = torch.nn.functional.softmax(
            attn_weights_17, dim=-1, dtype=torch.float32
        )
        attn_weights_17 = None
        attention_probs_16 = softmax_8.to(torch.bfloat16)
        softmax_8 = None
        attention_probs_17 = torch.nn.functional.dropout(
            attention_probs_16, 0.0, False, False
        )
        attention_probs_16 = None
        attention_probs_reshaped_8 = attention_probs_17.view(16, 9, -1)
        attention_probs_17 = None
        context_layer_16 = torch.bmm(attention_probs_reshaped_8, value_layer_17)
        attention_probs_reshaped_8 = value_layer_17 = None
        x_16 = context_layer_16.view(1, 16, 9, 64)
        context_layer_16 = None
        x_17 = x_16.permute(0, 2, 1, 3)
        x_16 = None
        context_layer_17 = x_17.reshape(1, 9, 1024)
        x_17 = None
        output_tensor_8 = torch._C._nn.linear(
            context_layer_17,
            l_self_modules_h_modules_8_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_8_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_17 = l_self_modules_h_modules_8_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_8_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_32 = torch.nn.functional.dropout(output_tensor_8, p=0.0, training=False)
        output_tensor_8 = None
        out_33 = out_31 + out_32
        out_31 = out_32 = None
        layernorm_output_17 = torch.nn.functional.layer_norm(
            out_33,
            (1024,),
            l_self_modules_h_modules_8_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_8_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_8_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_8_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_34 = torch._C._nn.linear(
            layernorm_output_17,
            l_self_modules_h_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_17 = l_self_modules_h_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_50 = linear_34 * 0.5
        mul_51 = 0.79788456 * linear_34
        mul_52 = 0.044715 * linear_34
        mul_53 = mul_52 * linear_34
        mul_52 = linear_34 = None
        add_43 = 1 + mul_53
        mul_53 = None
        mul_54 = mul_51 * add_43
        mul_51 = add_43 = None
        tanh_8 = torch.tanh(mul_54)
        mul_54 = None
        add_44 = 1.0 + tanh_8
        tanh_8 = None
        hidden_states_9 = mul_50 * add_44
        mul_50 = add_44 = None
        intermediate_output_8 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_h_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_9 = l_self_modules_h_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_34 = torch.nn.functional.dropout(
            intermediate_output_8, p=0.0, training=False
        )
        intermediate_output_8 = None
        out_35 = out_33 + out_34
        out_33 = out_34 = None
        layernorm_output_18 = torch.nn.functional.layer_norm(
            out_35,
            (1024,),
            l_self_modules_h_modules_9_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_9_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_9_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_9_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_18 = torch._C._nn.linear(
            layernorm_output_18,
            l_self_modules_h_modules_9_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_9_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_18 = l_self_modules_h_modules_9_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_9_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_19 = fused_qkv_18.view(1, 9, 16, 3, 64)
        fused_qkv_18 = None
        getitem_42 = fused_qkv_19[(Ellipsis, 0, slice(None, None, None))]
        query_layer_18 = getitem_42.transpose(1, 2)
        getitem_42 = None
        getitem_43 = fused_qkv_19[(Ellipsis, 1, slice(None, None, None))]
        key_layer_18 = getitem_43.transpose(1, 2)
        getitem_43 = None
        getitem_44 = fused_qkv_19[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_19 = None
        value_layer_18 = getitem_44.transpose(1, 2)
        getitem_44 = None
        query_layer_19 = query_layer_18.reshape(16, -1, 64)
        query_layer_18 = None
        reshape_39 = key_layer_18.reshape(16, -1, 64)
        key_layer_19 = reshape_39.transpose(-1, -2)
        reshape_39 = None
        value_layer_19 = value_layer_18.reshape(16, -1, 64)
        attention_scores_9 = alibi_1.baddbmm(
            batch1=query_layer_19, batch2=key_layer_19, beta=1.0, alpha=0.125
        )
        query_layer_19 = key_layer_19 = None
        attn_weights_18 = attention_scores_9.view(1, 16, 9, -1)
        attention_scores_9 = None
        causal_mask_14 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_19 = attn_weights_18 + causal_mask_14
        attn_weights_18 = causal_mask_14 = None
        softmax_9 = torch.nn.functional.softmax(
            attn_weights_19, dim=-1, dtype=torch.float32
        )
        attn_weights_19 = None
        attention_probs_18 = softmax_9.to(torch.bfloat16)
        softmax_9 = None
        attention_probs_19 = torch.nn.functional.dropout(
            attention_probs_18, 0.0, False, False
        )
        attention_probs_18 = None
        attention_probs_reshaped_9 = attention_probs_19.view(16, 9, -1)
        attention_probs_19 = None
        context_layer_18 = torch.bmm(attention_probs_reshaped_9, value_layer_19)
        attention_probs_reshaped_9 = value_layer_19 = None
        x_18 = context_layer_18.view(1, 16, 9, 64)
        context_layer_18 = None
        x_19 = x_18.permute(0, 2, 1, 3)
        x_18 = None
        context_layer_19 = x_19.reshape(1, 9, 1024)
        x_19 = None
        output_tensor_9 = torch._C._nn.linear(
            context_layer_19,
            l_self_modules_h_modules_9_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_9_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_19 = l_self_modules_h_modules_9_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_9_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_36 = torch.nn.functional.dropout(output_tensor_9, p=0.0, training=False)
        output_tensor_9 = None
        out_37 = out_35 + out_36
        out_35 = out_36 = None
        layernorm_output_19 = torch.nn.functional.layer_norm(
            out_37,
            (1024,),
            l_self_modules_h_modules_9_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_9_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_9_modules_post_attention_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_9_modules_post_attention_layernorm_parameters_bias_
        ) = None
        linear_38 = torch._C._nn.linear(
            layernorm_output_19,
            l_self_modules_h_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_19 = l_self_modules_h_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_56 = linear_38 * 0.5
        mul_57 = 0.79788456 * linear_38
        mul_58 = 0.044715 * linear_38
        mul_59 = mul_58 * linear_38
        mul_58 = linear_38 = None
        add_48 = 1 + mul_59
        mul_59 = None
        mul_60 = mul_57 * add_48
        mul_57 = add_48 = None
        tanh_9 = torch.tanh(mul_60)
        mul_60 = None
        add_49 = 1.0 + tanh_9
        tanh_9 = None
        hidden_states_10 = mul_56 * add_49
        mul_56 = add_49 = None
        intermediate_output_9 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_h_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_10 = l_self_modules_h_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_38 = torch.nn.functional.dropout(
            intermediate_output_9, p=0.0, training=False
        )
        intermediate_output_9 = None
        out_39 = out_37 + out_38
        out_37 = out_38 = None
        layernorm_output_20 = torch.nn.functional.layer_norm(
            out_39,
            (1024,),
            l_self_modules_h_modules_10_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_10_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_10_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_10_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_20 = torch._C._nn.linear(
            layernorm_output_20,
            l_self_modules_h_modules_10_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_10_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_20 = l_self_modules_h_modules_10_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_10_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_21 = fused_qkv_20.view(1, 9, 16, 3, 64)
        fused_qkv_20 = None
        getitem_46 = fused_qkv_21[(Ellipsis, 0, slice(None, None, None))]
        query_layer_20 = getitem_46.transpose(1, 2)
        getitem_46 = None
        getitem_47 = fused_qkv_21[(Ellipsis, 1, slice(None, None, None))]
        key_layer_20 = getitem_47.transpose(1, 2)
        getitem_47 = None
        getitem_48 = fused_qkv_21[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_21 = None
        value_layer_20 = getitem_48.transpose(1, 2)
        getitem_48 = None
        query_layer_21 = query_layer_20.reshape(16, -1, 64)
        query_layer_20 = None
        reshape_43 = key_layer_20.reshape(16, -1, 64)
        key_layer_21 = reshape_43.transpose(-1, -2)
        reshape_43 = None
        value_layer_21 = value_layer_20.reshape(16, -1, 64)
        attention_scores_10 = alibi_1.baddbmm(
            batch1=query_layer_21, batch2=key_layer_21, beta=1.0, alpha=0.125
        )
        query_layer_21 = key_layer_21 = None
        attn_weights_20 = attention_scores_10.view(1, 16, 9, -1)
        attention_scores_10 = None
        causal_mask_15 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_21 = attn_weights_20 + causal_mask_15
        attn_weights_20 = causal_mask_15 = None
        softmax_10 = torch.nn.functional.softmax(
            attn_weights_21, dim=-1, dtype=torch.float32
        )
        attn_weights_21 = None
        attention_probs_20 = softmax_10.to(torch.bfloat16)
        softmax_10 = None
        attention_probs_21 = torch.nn.functional.dropout(
            attention_probs_20, 0.0, False, False
        )
        attention_probs_20 = None
        attention_probs_reshaped_10 = attention_probs_21.view(16, 9, -1)
        attention_probs_21 = None
        context_layer_20 = torch.bmm(attention_probs_reshaped_10, value_layer_21)
        attention_probs_reshaped_10 = value_layer_21 = None
        x_20 = context_layer_20.view(1, 16, 9, 64)
        context_layer_20 = None
        x_21 = x_20.permute(0, 2, 1, 3)
        x_20 = None
        context_layer_21 = x_21.reshape(1, 9, 1024)
        x_21 = None
        output_tensor_10 = torch._C._nn.linear(
            context_layer_21,
            l_self_modules_h_modules_10_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_10_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_21 = l_self_modules_h_modules_10_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_10_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_40 = torch.nn.functional.dropout(output_tensor_10, p=0.0, training=False)
        output_tensor_10 = None
        out_41 = out_39 + out_40
        out_39 = out_40 = None
        layernorm_output_21 = torch.nn.functional.layer_norm(
            out_41,
            (1024,),
            l_self_modules_h_modules_10_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_10_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_10_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_h_modules_10_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_42 = torch._C._nn.linear(
            layernorm_output_21,
            l_self_modules_h_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_21 = l_self_modules_h_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_62 = linear_42 * 0.5
        mul_63 = 0.79788456 * linear_42
        mul_64 = 0.044715 * linear_42
        mul_65 = mul_64 * linear_42
        mul_64 = linear_42 = None
        add_53 = 1 + mul_65
        mul_65 = None
        mul_66 = mul_63 * add_53
        mul_63 = add_53 = None
        tanh_10 = torch.tanh(mul_66)
        mul_66 = None
        add_54 = 1.0 + tanh_10
        tanh_10 = None
        hidden_states_11 = mul_62 * add_54
        mul_62 = add_54 = None
        intermediate_output_10 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_h_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_11 = l_self_modules_h_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_42 = torch.nn.functional.dropout(
            intermediate_output_10, p=0.0, training=False
        )
        intermediate_output_10 = None
        out_43 = out_41 + out_42
        out_41 = out_42 = None
        layernorm_output_22 = torch.nn.functional.layer_norm(
            out_43,
            (1024,),
            l_self_modules_h_modules_11_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_11_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_11_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_11_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_22 = torch._C._nn.linear(
            layernorm_output_22,
            l_self_modules_h_modules_11_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_11_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_22 = l_self_modules_h_modules_11_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_11_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_23 = fused_qkv_22.view(1, 9, 16, 3, 64)
        fused_qkv_22 = None
        getitem_50 = fused_qkv_23[(Ellipsis, 0, slice(None, None, None))]
        query_layer_22 = getitem_50.transpose(1, 2)
        getitem_50 = None
        getitem_51 = fused_qkv_23[(Ellipsis, 1, slice(None, None, None))]
        key_layer_22 = getitem_51.transpose(1, 2)
        getitem_51 = None
        getitem_52 = fused_qkv_23[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_23 = None
        value_layer_22 = getitem_52.transpose(1, 2)
        getitem_52 = None
        query_layer_23 = query_layer_22.reshape(16, -1, 64)
        query_layer_22 = None
        reshape_47 = key_layer_22.reshape(16, -1, 64)
        key_layer_23 = reshape_47.transpose(-1, -2)
        reshape_47 = None
        value_layer_23 = value_layer_22.reshape(16, -1, 64)
        attention_scores_11 = alibi_1.baddbmm(
            batch1=query_layer_23, batch2=key_layer_23, beta=1.0, alpha=0.125
        )
        query_layer_23 = key_layer_23 = None
        attn_weights_22 = attention_scores_11.view(1, 16, 9, -1)
        attention_scores_11 = None
        causal_mask_16 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_23 = attn_weights_22 + causal_mask_16
        attn_weights_22 = causal_mask_16 = None
        softmax_11 = torch.nn.functional.softmax(
            attn_weights_23, dim=-1, dtype=torch.float32
        )
        attn_weights_23 = None
        attention_probs_22 = softmax_11.to(torch.bfloat16)
        softmax_11 = None
        attention_probs_23 = torch.nn.functional.dropout(
            attention_probs_22, 0.0, False, False
        )
        attention_probs_22 = None
        attention_probs_reshaped_11 = attention_probs_23.view(16, 9, -1)
        attention_probs_23 = None
        context_layer_22 = torch.bmm(attention_probs_reshaped_11, value_layer_23)
        attention_probs_reshaped_11 = value_layer_23 = None
        x_22 = context_layer_22.view(1, 16, 9, 64)
        context_layer_22 = None
        x_23 = x_22.permute(0, 2, 1, 3)
        x_22 = None
        context_layer_23 = x_23.reshape(1, 9, 1024)
        x_23 = None
        output_tensor_11 = torch._C._nn.linear(
            context_layer_23,
            l_self_modules_h_modules_11_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_11_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_23 = l_self_modules_h_modules_11_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_11_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_44 = torch.nn.functional.dropout(output_tensor_11, p=0.0, training=False)
        output_tensor_11 = None
        out_45 = out_43 + out_44
        out_43 = out_44 = None
        layernorm_output_23 = torch.nn.functional.layer_norm(
            out_45,
            (1024,),
            l_self_modules_h_modules_11_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_11_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_11_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_h_modules_11_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_46 = torch._C._nn.linear(
            layernorm_output_23,
            l_self_modules_h_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_23 = l_self_modules_h_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_68 = linear_46 * 0.5
        mul_69 = 0.79788456 * linear_46
        mul_70 = 0.044715 * linear_46
        mul_71 = mul_70 * linear_46
        mul_70 = linear_46 = None
        add_58 = 1 + mul_71
        mul_71 = None
        mul_72 = mul_69 * add_58
        mul_69 = add_58 = None
        tanh_11 = torch.tanh(mul_72)
        mul_72 = None
        add_59 = 1.0 + tanh_11
        tanh_11 = None
        hidden_states_12 = mul_68 * add_59
        mul_68 = add_59 = None
        intermediate_output_11 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_h_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_12 = l_self_modules_h_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_46 = torch.nn.functional.dropout(
            intermediate_output_11, p=0.0, training=False
        )
        intermediate_output_11 = None
        out_47 = out_45 + out_46
        out_45 = out_46 = None
        layernorm_output_24 = torch.nn.functional.layer_norm(
            out_47,
            (1024,),
            l_self_modules_h_modules_12_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_12_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_12_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_12_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_24 = torch._C._nn.linear(
            layernorm_output_24,
            l_self_modules_h_modules_12_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_12_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_24 = l_self_modules_h_modules_12_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_12_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_25 = fused_qkv_24.view(1, 9, 16, 3, 64)
        fused_qkv_24 = None
        getitem_54 = fused_qkv_25[(Ellipsis, 0, slice(None, None, None))]
        query_layer_24 = getitem_54.transpose(1, 2)
        getitem_54 = None
        getitem_55 = fused_qkv_25[(Ellipsis, 1, slice(None, None, None))]
        key_layer_24 = getitem_55.transpose(1, 2)
        getitem_55 = None
        getitem_56 = fused_qkv_25[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_25 = None
        value_layer_24 = getitem_56.transpose(1, 2)
        getitem_56 = None
        query_layer_25 = query_layer_24.reshape(16, -1, 64)
        query_layer_24 = None
        reshape_51 = key_layer_24.reshape(16, -1, 64)
        key_layer_25 = reshape_51.transpose(-1, -2)
        reshape_51 = None
        value_layer_25 = value_layer_24.reshape(16, -1, 64)
        attention_scores_12 = alibi_1.baddbmm(
            batch1=query_layer_25, batch2=key_layer_25, beta=1.0, alpha=0.125
        )
        query_layer_25 = key_layer_25 = None
        attn_weights_24 = attention_scores_12.view(1, 16, 9, -1)
        attention_scores_12 = None
        causal_mask_17 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_25 = attn_weights_24 + causal_mask_17
        attn_weights_24 = causal_mask_17 = None
        softmax_12 = torch.nn.functional.softmax(
            attn_weights_25, dim=-1, dtype=torch.float32
        )
        attn_weights_25 = None
        attention_probs_24 = softmax_12.to(torch.bfloat16)
        softmax_12 = None
        attention_probs_25 = torch.nn.functional.dropout(
            attention_probs_24, 0.0, False, False
        )
        attention_probs_24 = None
        attention_probs_reshaped_12 = attention_probs_25.view(16, 9, -1)
        attention_probs_25 = None
        context_layer_24 = torch.bmm(attention_probs_reshaped_12, value_layer_25)
        attention_probs_reshaped_12 = value_layer_25 = None
        x_24 = context_layer_24.view(1, 16, 9, 64)
        context_layer_24 = None
        x_25 = x_24.permute(0, 2, 1, 3)
        x_24 = None
        context_layer_25 = x_25.reshape(1, 9, 1024)
        x_25 = None
        output_tensor_12 = torch._C._nn.linear(
            context_layer_25,
            l_self_modules_h_modules_12_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_12_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_25 = l_self_modules_h_modules_12_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_12_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_48 = torch.nn.functional.dropout(output_tensor_12, p=0.0, training=False)
        output_tensor_12 = None
        out_49 = out_47 + out_48
        out_47 = out_48 = None
        layernorm_output_25 = torch.nn.functional.layer_norm(
            out_49,
            (1024,),
            l_self_modules_h_modules_12_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_12_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_12_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_h_modules_12_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_50 = torch._C._nn.linear(
            layernorm_output_25,
            l_self_modules_h_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_25 = l_self_modules_h_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_74 = linear_50 * 0.5
        mul_75 = 0.79788456 * linear_50
        mul_76 = 0.044715 * linear_50
        mul_77 = mul_76 * linear_50
        mul_76 = linear_50 = None
        add_63 = 1 + mul_77
        mul_77 = None
        mul_78 = mul_75 * add_63
        mul_75 = add_63 = None
        tanh_12 = torch.tanh(mul_78)
        mul_78 = None
        add_64 = 1.0 + tanh_12
        tanh_12 = None
        hidden_states_13 = mul_74 * add_64
        mul_74 = add_64 = None
        intermediate_output_12 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_h_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_13 = l_self_modules_h_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_50 = torch.nn.functional.dropout(
            intermediate_output_12, p=0.0, training=False
        )
        intermediate_output_12 = None
        out_51 = out_49 + out_50
        out_49 = out_50 = None
        layernorm_output_26 = torch.nn.functional.layer_norm(
            out_51,
            (1024,),
            l_self_modules_h_modules_13_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_13_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_13_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_13_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_26 = torch._C._nn.linear(
            layernorm_output_26,
            l_self_modules_h_modules_13_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_13_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_26 = l_self_modules_h_modules_13_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_13_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_27 = fused_qkv_26.view(1, 9, 16, 3, 64)
        fused_qkv_26 = None
        getitem_58 = fused_qkv_27[(Ellipsis, 0, slice(None, None, None))]
        query_layer_26 = getitem_58.transpose(1, 2)
        getitem_58 = None
        getitem_59 = fused_qkv_27[(Ellipsis, 1, slice(None, None, None))]
        key_layer_26 = getitem_59.transpose(1, 2)
        getitem_59 = None
        getitem_60 = fused_qkv_27[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_27 = None
        value_layer_26 = getitem_60.transpose(1, 2)
        getitem_60 = None
        query_layer_27 = query_layer_26.reshape(16, -1, 64)
        query_layer_26 = None
        reshape_55 = key_layer_26.reshape(16, -1, 64)
        key_layer_27 = reshape_55.transpose(-1, -2)
        reshape_55 = None
        value_layer_27 = value_layer_26.reshape(16, -1, 64)
        attention_scores_13 = alibi_1.baddbmm(
            batch1=query_layer_27, batch2=key_layer_27, beta=1.0, alpha=0.125
        )
        query_layer_27 = key_layer_27 = None
        attn_weights_26 = attention_scores_13.view(1, 16, 9, -1)
        attention_scores_13 = None
        causal_mask_18 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_27 = attn_weights_26 + causal_mask_18
        attn_weights_26 = causal_mask_18 = None
        softmax_13 = torch.nn.functional.softmax(
            attn_weights_27, dim=-1, dtype=torch.float32
        )
        attn_weights_27 = None
        attention_probs_26 = softmax_13.to(torch.bfloat16)
        softmax_13 = None
        attention_probs_27 = torch.nn.functional.dropout(
            attention_probs_26, 0.0, False, False
        )
        attention_probs_26 = None
        attention_probs_reshaped_13 = attention_probs_27.view(16, 9, -1)
        attention_probs_27 = None
        context_layer_26 = torch.bmm(attention_probs_reshaped_13, value_layer_27)
        attention_probs_reshaped_13 = value_layer_27 = None
        x_26 = context_layer_26.view(1, 16, 9, 64)
        context_layer_26 = None
        x_27 = x_26.permute(0, 2, 1, 3)
        x_26 = None
        context_layer_27 = x_27.reshape(1, 9, 1024)
        x_27 = None
        output_tensor_13 = torch._C._nn.linear(
            context_layer_27,
            l_self_modules_h_modules_13_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_13_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_27 = l_self_modules_h_modules_13_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_13_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_52 = torch.nn.functional.dropout(output_tensor_13, p=0.0, training=False)
        output_tensor_13 = None
        out_53 = out_51 + out_52
        out_51 = out_52 = None
        layernorm_output_27 = torch.nn.functional.layer_norm(
            out_53,
            (1024,),
            l_self_modules_h_modules_13_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_13_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_13_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_h_modules_13_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            layernorm_output_27,
            l_self_modules_h_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_27 = l_self_modules_h_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_80 = linear_54 * 0.5
        mul_81 = 0.79788456 * linear_54
        mul_82 = 0.044715 * linear_54
        mul_83 = mul_82 * linear_54
        mul_82 = linear_54 = None
        add_68 = 1 + mul_83
        mul_83 = None
        mul_84 = mul_81 * add_68
        mul_81 = add_68 = None
        tanh_13 = torch.tanh(mul_84)
        mul_84 = None
        add_69 = 1.0 + tanh_13
        tanh_13 = None
        hidden_states_14 = mul_80 * add_69
        mul_80 = add_69 = None
        intermediate_output_13 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_h_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_14 = l_self_modules_h_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_54 = torch.nn.functional.dropout(
            intermediate_output_13, p=0.0, training=False
        )
        intermediate_output_13 = None
        out_55 = out_53 + out_54
        out_53 = out_54 = None
        layernorm_output_28 = torch.nn.functional.layer_norm(
            out_55,
            (1024,),
            l_self_modules_h_modules_14_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_14_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_14_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_14_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_28 = torch._C._nn.linear(
            layernorm_output_28,
            l_self_modules_h_modules_14_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_14_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_28 = l_self_modules_h_modules_14_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_14_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_29 = fused_qkv_28.view(1, 9, 16, 3, 64)
        fused_qkv_28 = None
        getitem_62 = fused_qkv_29[(Ellipsis, 0, slice(None, None, None))]
        query_layer_28 = getitem_62.transpose(1, 2)
        getitem_62 = None
        getitem_63 = fused_qkv_29[(Ellipsis, 1, slice(None, None, None))]
        key_layer_28 = getitem_63.transpose(1, 2)
        getitem_63 = None
        getitem_64 = fused_qkv_29[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_29 = None
        value_layer_28 = getitem_64.transpose(1, 2)
        getitem_64 = None
        query_layer_29 = query_layer_28.reshape(16, -1, 64)
        query_layer_28 = None
        reshape_59 = key_layer_28.reshape(16, -1, 64)
        key_layer_29 = reshape_59.transpose(-1, -2)
        reshape_59 = None
        value_layer_29 = value_layer_28.reshape(16, -1, 64)
        attention_scores_14 = alibi_1.baddbmm(
            batch1=query_layer_29, batch2=key_layer_29, beta=1.0, alpha=0.125
        )
        query_layer_29 = key_layer_29 = None
        attn_weights_28 = attention_scores_14.view(1, 16, 9, -1)
        attention_scores_14 = None
        causal_mask_19 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_29 = attn_weights_28 + causal_mask_19
        attn_weights_28 = causal_mask_19 = None
        softmax_14 = torch.nn.functional.softmax(
            attn_weights_29, dim=-1, dtype=torch.float32
        )
        attn_weights_29 = None
        attention_probs_28 = softmax_14.to(torch.bfloat16)
        softmax_14 = None
        attention_probs_29 = torch.nn.functional.dropout(
            attention_probs_28, 0.0, False, False
        )
        attention_probs_28 = None
        attention_probs_reshaped_14 = attention_probs_29.view(16, 9, -1)
        attention_probs_29 = None
        context_layer_28 = torch.bmm(attention_probs_reshaped_14, value_layer_29)
        attention_probs_reshaped_14 = value_layer_29 = None
        x_28 = context_layer_28.view(1, 16, 9, 64)
        context_layer_28 = None
        x_29 = x_28.permute(0, 2, 1, 3)
        x_28 = None
        context_layer_29 = x_29.reshape(1, 9, 1024)
        x_29 = None
        output_tensor_14 = torch._C._nn.linear(
            context_layer_29,
            l_self_modules_h_modules_14_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_14_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_29 = l_self_modules_h_modules_14_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_14_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_56 = torch.nn.functional.dropout(output_tensor_14, p=0.0, training=False)
        output_tensor_14 = None
        out_57 = out_55 + out_56
        out_55 = out_56 = None
        layernorm_output_29 = torch.nn.functional.layer_norm(
            out_57,
            (1024,),
            l_self_modules_h_modules_14_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_14_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_14_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_h_modules_14_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_58 = torch._C._nn.linear(
            layernorm_output_29,
            l_self_modules_h_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_29 = l_self_modules_h_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_86 = linear_58 * 0.5
        mul_87 = 0.79788456 * linear_58
        mul_88 = 0.044715 * linear_58
        mul_89 = mul_88 * linear_58
        mul_88 = linear_58 = None
        add_73 = 1 + mul_89
        mul_89 = None
        mul_90 = mul_87 * add_73
        mul_87 = add_73 = None
        tanh_14 = torch.tanh(mul_90)
        mul_90 = None
        add_74 = 1.0 + tanh_14
        tanh_14 = None
        hidden_states_15 = mul_86 * add_74
        mul_86 = add_74 = None
        intermediate_output_14 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_h_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_15 = l_self_modules_h_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_58 = torch.nn.functional.dropout(
            intermediate_output_14, p=0.0, training=False
        )
        intermediate_output_14 = None
        out_59 = out_57 + out_58
        out_57 = out_58 = None
        layernorm_output_30 = torch.nn.functional.layer_norm(
            out_59,
            (1024,),
            l_self_modules_h_modules_15_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_15_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_15_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_15_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_30 = torch._C._nn.linear(
            layernorm_output_30,
            l_self_modules_h_modules_15_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_15_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_30 = l_self_modules_h_modules_15_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_15_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_31 = fused_qkv_30.view(1, 9, 16, 3, 64)
        fused_qkv_30 = None
        getitem_66 = fused_qkv_31[(Ellipsis, 0, slice(None, None, None))]
        query_layer_30 = getitem_66.transpose(1, 2)
        getitem_66 = None
        getitem_67 = fused_qkv_31[(Ellipsis, 1, slice(None, None, None))]
        key_layer_30 = getitem_67.transpose(1, 2)
        getitem_67 = None
        getitem_68 = fused_qkv_31[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_31 = None
        value_layer_30 = getitem_68.transpose(1, 2)
        getitem_68 = None
        query_layer_31 = query_layer_30.reshape(16, -1, 64)
        query_layer_30 = None
        reshape_63 = key_layer_30.reshape(16, -1, 64)
        key_layer_31 = reshape_63.transpose(-1, -2)
        reshape_63 = None
        value_layer_31 = value_layer_30.reshape(16, -1, 64)
        attention_scores_15 = alibi_1.baddbmm(
            batch1=query_layer_31, batch2=key_layer_31, beta=1.0, alpha=0.125
        )
        query_layer_31 = key_layer_31 = None
        attn_weights_30 = attention_scores_15.view(1, 16, 9, -1)
        attention_scores_15 = None
        causal_mask_20 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_31 = attn_weights_30 + causal_mask_20
        attn_weights_30 = causal_mask_20 = None
        softmax_15 = torch.nn.functional.softmax(
            attn_weights_31, dim=-1, dtype=torch.float32
        )
        attn_weights_31 = None
        attention_probs_30 = softmax_15.to(torch.bfloat16)
        softmax_15 = None
        attention_probs_31 = torch.nn.functional.dropout(
            attention_probs_30, 0.0, False, False
        )
        attention_probs_30 = None
        attention_probs_reshaped_15 = attention_probs_31.view(16, 9, -1)
        attention_probs_31 = None
        context_layer_30 = torch.bmm(attention_probs_reshaped_15, value_layer_31)
        attention_probs_reshaped_15 = value_layer_31 = None
        x_30 = context_layer_30.view(1, 16, 9, 64)
        context_layer_30 = None
        x_31 = x_30.permute(0, 2, 1, 3)
        x_30 = None
        context_layer_31 = x_31.reshape(1, 9, 1024)
        x_31 = None
        output_tensor_15 = torch._C._nn.linear(
            context_layer_31,
            l_self_modules_h_modules_15_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_15_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_31 = l_self_modules_h_modules_15_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_15_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_60 = torch.nn.functional.dropout(output_tensor_15, p=0.0, training=False)
        output_tensor_15 = None
        out_61 = out_59 + out_60
        out_59 = out_60 = None
        layernorm_output_31 = torch.nn.functional.layer_norm(
            out_61,
            (1024,),
            l_self_modules_h_modules_15_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_15_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_15_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_h_modules_15_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_62 = torch._C._nn.linear(
            layernorm_output_31,
            l_self_modules_h_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_31 = l_self_modules_h_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_92 = linear_62 * 0.5
        mul_93 = 0.79788456 * linear_62
        mul_94 = 0.044715 * linear_62
        mul_95 = mul_94 * linear_62
        mul_94 = linear_62 = None
        add_78 = 1 + mul_95
        mul_95 = None
        mul_96 = mul_93 * add_78
        mul_93 = add_78 = None
        tanh_15 = torch.tanh(mul_96)
        mul_96 = None
        add_79 = 1.0 + tanh_15
        tanh_15 = None
        hidden_states_16 = mul_92 * add_79
        mul_92 = add_79 = None
        intermediate_output_15 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_h_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_16 = l_self_modules_h_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_62 = torch.nn.functional.dropout(
            intermediate_output_15, p=0.0, training=False
        )
        intermediate_output_15 = None
        out_63 = out_61 + out_62
        out_61 = out_62 = None
        layernorm_output_32 = torch.nn.functional.layer_norm(
            out_63,
            (1024,),
            l_self_modules_h_modules_16_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_16_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_16_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_16_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_32 = torch._C._nn.linear(
            layernorm_output_32,
            l_self_modules_h_modules_16_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_16_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_32 = l_self_modules_h_modules_16_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_16_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_33 = fused_qkv_32.view(1, 9, 16, 3, 64)
        fused_qkv_32 = None
        getitem_70 = fused_qkv_33[(Ellipsis, 0, slice(None, None, None))]
        query_layer_32 = getitem_70.transpose(1, 2)
        getitem_70 = None
        getitem_71 = fused_qkv_33[(Ellipsis, 1, slice(None, None, None))]
        key_layer_32 = getitem_71.transpose(1, 2)
        getitem_71 = None
        getitem_72 = fused_qkv_33[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_33 = None
        value_layer_32 = getitem_72.transpose(1, 2)
        getitem_72 = None
        query_layer_33 = query_layer_32.reshape(16, -1, 64)
        query_layer_32 = None
        reshape_67 = key_layer_32.reshape(16, -1, 64)
        key_layer_33 = reshape_67.transpose(-1, -2)
        reshape_67 = None
        value_layer_33 = value_layer_32.reshape(16, -1, 64)
        attention_scores_16 = alibi_1.baddbmm(
            batch1=query_layer_33, batch2=key_layer_33, beta=1.0, alpha=0.125
        )
        query_layer_33 = key_layer_33 = None
        attn_weights_32 = attention_scores_16.view(1, 16, 9, -1)
        attention_scores_16 = None
        causal_mask_21 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_33 = attn_weights_32 + causal_mask_21
        attn_weights_32 = causal_mask_21 = None
        softmax_16 = torch.nn.functional.softmax(
            attn_weights_33, dim=-1, dtype=torch.float32
        )
        attn_weights_33 = None
        attention_probs_32 = softmax_16.to(torch.bfloat16)
        softmax_16 = None
        attention_probs_33 = torch.nn.functional.dropout(
            attention_probs_32, 0.0, False, False
        )
        attention_probs_32 = None
        attention_probs_reshaped_16 = attention_probs_33.view(16, 9, -1)
        attention_probs_33 = None
        context_layer_32 = torch.bmm(attention_probs_reshaped_16, value_layer_33)
        attention_probs_reshaped_16 = value_layer_33 = None
        x_32 = context_layer_32.view(1, 16, 9, 64)
        context_layer_32 = None
        x_33 = x_32.permute(0, 2, 1, 3)
        x_32 = None
        context_layer_33 = x_33.reshape(1, 9, 1024)
        x_33 = None
        output_tensor_16 = torch._C._nn.linear(
            context_layer_33,
            l_self_modules_h_modules_16_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_16_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_33 = l_self_modules_h_modules_16_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_16_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_64 = torch.nn.functional.dropout(output_tensor_16, p=0.0, training=False)
        output_tensor_16 = None
        out_65 = out_63 + out_64
        out_63 = out_64 = None
        layernorm_output_33 = torch.nn.functional.layer_norm(
            out_65,
            (1024,),
            l_self_modules_h_modules_16_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_16_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_16_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_h_modules_16_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_66 = torch._C._nn.linear(
            layernorm_output_33,
            l_self_modules_h_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_33 = l_self_modules_h_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_16_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_98 = linear_66 * 0.5
        mul_99 = 0.79788456 * linear_66
        mul_100 = 0.044715 * linear_66
        mul_101 = mul_100 * linear_66
        mul_100 = linear_66 = None
        add_83 = 1 + mul_101
        mul_101 = None
        mul_102 = mul_99 * add_83
        mul_99 = add_83 = None
        tanh_16 = torch.tanh(mul_102)
        mul_102 = None
        add_84 = 1.0 + tanh_16
        tanh_16 = None
        hidden_states_17 = mul_98 * add_84
        mul_98 = add_84 = None
        intermediate_output_16 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_h_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_17 = l_self_modules_h_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_16_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_66 = torch.nn.functional.dropout(
            intermediate_output_16, p=0.0, training=False
        )
        intermediate_output_16 = None
        out_67 = out_65 + out_66
        out_65 = out_66 = None
        layernorm_output_34 = torch.nn.functional.layer_norm(
            out_67,
            (1024,),
            l_self_modules_h_modules_17_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_17_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_17_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_17_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_34 = torch._C._nn.linear(
            layernorm_output_34,
            l_self_modules_h_modules_17_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_17_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_34 = l_self_modules_h_modules_17_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_17_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_35 = fused_qkv_34.view(1, 9, 16, 3, 64)
        fused_qkv_34 = None
        getitem_74 = fused_qkv_35[(Ellipsis, 0, slice(None, None, None))]
        query_layer_34 = getitem_74.transpose(1, 2)
        getitem_74 = None
        getitem_75 = fused_qkv_35[(Ellipsis, 1, slice(None, None, None))]
        key_layer_34 = getitem_75.transpose(1, 2)
        getitem_75 = None
        getitem_76 = fused_qkv_35[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_35 = None
        value_layer_34 = getitem_76.transpose(1, 2)
        getitem_76 = None
        query_layer_35 = query_layer_34.reshape(16, -1, 64)
        query_layer_34 = None
        reshape_71 = key_layer_34.reshape(16, -1, 64)
        key_layer_35 = reshape_71.transpose(-1, -2)
        reshape_71 = None
        value_layer_35 = value_layer_34.reshape(16, -1, 64)
        attention_scores_17 = alibi_1.baddbmm(
            batch1=query_layer_35, batch2=key_layer_35, beta=1.0, alpha=0.125
        )
        query_layer_35 = key_layer_35 = None
        attn_weights_34 = attention_scores_17.view(1, 16, 9, -1)
        attention_scores_17 = None
        causal_mask_22 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_35 = attn_weights_34 + causal_mask_22
        attn_weights_34 = causal_mask_22 = None
        softmax_17 = torch.nn.functional.softmax(
            attn_weights_35, dim=-1, dtype=torch.float32
        )
        attn_weights_35 = None
        attention_probs_34 = softmax_17.to(torch.bfloat16)
        softmax_17 = None
        attention_probs_35 = torch.nn.functional.dropout(
            attention_probs_34, 0.0, False, False
        )
        attention_probs_34 = None
        attention_probs_reshaped_17 = attention_probs_35.view(16, 9, -1)
        attention_probs_35 = None
        context_layer_34 = torch.bmm(attention_probs_reshaped_17, value_layer_35)
        attention_probs_reshaped_17 = value_layer_35 = None
        x_34 = context_layer_34.view(1, 16, 9, 64)
        context_layer_34 = None
        x_35 = x_34.permute(0, 2, 1, 3)
        x_34 = None
        context_layer_35 = x_35.reshape(1, 9, 1024)
        x_35 = None
        output_tensor_17 = torch._C._nn.linear(
            context_layer_35,
            l_self_modules_h_modules_17_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_17_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_35 = l_self_modules_h_modules_17_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_17_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_68 = torch.nn.functional.dropout(output_tensor_17, p=0.0, training=False)
        output_tensor_17 = None
        out_69 = out_67 + out_68
        out_67 = out_68 = None
        layernorm_output_35 = torch.nn.functional.layer_norm(
            out_69,
            (1024,),
            l_self_modules_h_modules_17_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_17_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_17_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_h_modules_17_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_70 = torch._C._nn.linear(
            layernorm_output_35,
            l_self_modules_h_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_35 = l_self_modules_h_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_17_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_104 = linear_70 * 0.5
        mul_105 = 0.79788456 * linear_70
        mul_106 = 0.044715 * linear_70
        mul_107 = mul_106 * linear_70
        mul_106 = linear_70 = None
        add_88 = 1 + mul_107
        mul_107 = None
        mul_108 = mul_105 * add_88
        mul_105 = add_88 = None
        tanh_17 = torch.tanh(mul_108)
        mul_108 = None
        add_89 = 1.0 + tanh_17
        tanh_17 = None
        hidden_states_18 = mul_104 * add_89
        mul_104 = add_89 = None
        intermediate_output_17 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_h_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_18 = l_self_modules_h_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_17_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_70 = torch.nn.functional.dropout(
            intermediate_output_17, p=0.0, training=False
        )
        intermediate_output_17 = None
        out_71 = out_69 + out_70
        out_69 = out_70 = None
        layernorm_output_36 = torch.nn.functional.layer_norm(
            out_71,
            (1024,),
            l_self_modules_h_modules_18_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_18_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_18_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_18_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_36 = torch._C._nn.linear(
            layernorm_output_36,
            l_self_modules_h_modules_18_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_18_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_36 = l_self_modules_h_modules_18_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_18_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_37 = fused_qkv_36.view(1, 9, 16, 3, 64)
        fused_qkv_36 = None
        getitem_78 = fused_qkv_37[(Ellipsis, 0, slice(None, None, None))]
        query_layer_36 = getitem_78.transpose(1, 2)
        getitem_78 = None
        getitem_79 = fused_qkv_37[(Ellipsis, 1, slice(None, None, None))]
        key_layer_36 = getitem_79.transpose(1, 2)
        getitem_79 = None
        getitem_80 = fused_qkv_37[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_37 = None
        value_layer_36 = getitem_80.transpose(1, 2)
        getitem_80 = None
        query_layer_37 = query_layer_36.reshape(16, -1, 64)
        query_layer_36 = None
        reshape_75 = key_layer_36.reshape(16, -1, 64)
        key_layer_37 = reshape_75.transpose(-1, -2)
        reshape_75 = None
        value_layer_37 = value_layer_36.reshape(16, -1, 64)
        attention_scores_18 = alibi_1.baddbmm(
            batch1=query_layer_37, batch2=key_layer_37, beta=1.0, alpha=0.125
        )
        query_layer_37 = key_layer_37 = None
        attn_weights_36 = attention_scores_18.view(1, 16, 9, -1)
        attention_scores_18 = None
        causal_mask_23 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_37 = attn_weights_36 + causal_mask_23
        attn_weights_36 = causal_mask_23 = None
        softmax_18 = torch.nn.functional.softmax(
            attn_weights_37, dim=-1, dtype=torch.float32
        )
        attn_weights_37 = None
        attention_probs_36 = softmax_18.to(torch.bfloat16)
        softmax_18 = None
        attention_probs_37 = torch.nn.functional.dropout(
            attention_probs_36, 0.0, False, False
        )
        attention_probs_36 = None
        attention_probs_reshaped_18 = attention_probs_37.view(16, 9, -1)
        attention_probs_37 = None
        context_layer_36 = torch.bmm(attention_probs_reshaped_18, value_layer_37)
        attention_probs_reshaped_18 = value_layer_37 = None
        x_36 = context_layer_36.view(1, 16, 9, 64)
        context_layer_36 = None
        x_37 = x_36.permute(0, 2, 1, 3)
        x_36 = None
        context_layer_37 = x_37.reshape(1, 9, 1024)
        x_37 = None
        output_tensor_18 = torch._C._nn.linear(
            context_layer_37,
            l_self_modules_h_modules_18_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_18_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_37 = l_self_modules_h_modules_18_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_18_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_72 = torch.nn.functional.dropout(output_tensor_18, p=0.0, training=False)
        output_tensor_18 = None
        out_73 = out_71 + out_72
        out_71 = out_72 = None
        layernorm_output_37 = torch.nn.functional.layer_norm(
            out_73,
            (1024,),
            l_self_modules_h_modules_18_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_18_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_18_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_h_modules_18_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_74 = torch._C._nn.linear(
            layernorm_output_37,
            l_self_modules_h_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_37 = l_self_modules_h_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_18_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_110 = linear_74 * 0.5
        mul_111 = 0.79788456 * linear_74
        mul_112 = 0.044715 * linear_74
        mul_113 = mul_112 * linear_74
        mul_112 = linear_74 = None
        add_93 = 1 + mul_113
        mul_113 = None
        mul_114 = mul_111 * add_93
        mul_111 = add_93 = None
        tanh_18 = torch.tanh(mul_114)
        mul_114 = None
        add_94 = 1.0 + tanh_18
        tanh_18 = None
        hidden_states_19 = mul_110 * add_94
        mul_110 = add_94 = None
        intermediate_output_18 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_h_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_19 = l_self_modules_h_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_18_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_74 = torch.nn.functional.dropout(
            intermediate_output_18, p=0.0, training=False
        )
        intermediate_output_18 = None
        out_75 = out_73 + out_74
        out_73 = out_74 = None
        layernorm_output_38 = torch.nn.functional.layer_norm(
            out_75,
            (1024,),
            l_self_modules_h_modules_19_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_19_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_19_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_19_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_38 = torch._C._nn.linear(
            layernorm_output_38,
            l_self_modules_h_modules_19_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_19_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_38 = l_self_modules_h_modules_19_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_19_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_39 = fused_qkv_38.view(1, 9, 16, 3, 64)
        fused_qkv_38 = None
        getitem_82 = fused_qkv_39[(Ellipsis, 0, slice(None, None, None))]
        query_layer_38 = getitem_82.transpose(1, 2)
        getitem_82 = None
        getitem_83 = fused_qkv_39[(Ellipsis, 1, slice(None, None, None))]
        key_layer_38 = getitem_83.transpose(1, 2)
        getitem_83 = None
        getitem_84 = fused_qkv_39[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_39 = None
        value_layer_38 = getitem_84.transpose(1, 2)
        getitem_84 = None
        query_layer_39 = query_layer_38.reshape(16, -1, 64)
        query_layer_38 = None
        reshape_79 = key_layer_38.reshape(16, -1, 64)
        key_layer_39 = reshape_79.transpose(-1, -2)
        reshape_79 = None
        value_layer_39 = value_layer_38.reshape(16, -1, 64)
        attention_scores_19 = alibi_1.baddbmm(
            batch1=query_layer_39, batch2=key_layer_39, beta=1.0, alpha=0.125
        )
        query_layer_39 = key_layer_39 = None
        attn_weights_38 = attention_scores_19.view(1, 16, 9, -1)
        attention_scores_19 = None
        causal_mask_24 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_39 = attn_weights_38 + causal_mask_24
        attn_weights_38 = causal_mask_24 = None
        softmax_19 = torch.nn.functional.softmax(
            attn_weights_39, dim=-1, dtype=torch.float32
        )
        attn_weights_39 = None
        attention_probs_38 = softmax_19.to(torch.bfloat16)
        softmax_19 = None
        attention_probs_39 = torch.nn.functional.dropout(
            attention_probs_38, 0.0, False, False
        )
        attention_probs_38 = None
        attention_probs_reshaped_19 = attention_probs_39.view(16, 9, -1)
        attention_probs_39 = None
        context_layer_38 = torch.bmm(attention_probs_reshaped_19, value_layer_39)
        attention_probs_reshaped_19 = value_layer_39 = None
        x_38 = context_layer_38.view(1, 16, 9, 64)
        context_layer_38 = None
        x_39 = x_38.permute(0, 2, 1, 3)
        x_38 = None
        context_layer_39 = x_39.reshape(1, 9, 1024)
        x_39 = None
        output_tensor_19 = torch._C._nn.linear(
            context_layer_39,
            l_self_modules_h_modules_19_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_19_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_39 = l_self_modules_h_modules_19_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_19_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_76 = torch.nn.functional.dropout(output_tensor_19, p=0.0, training=False)
        output_tensor_19 = None
        out_77 = out_75 + out_76
        out_75 = out_76 = None
        layernorm_output_39 = torch.nn.functional.layer_norm(
            out_77,
            (1024,),
            l_self_modules_h_modules_19_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_19_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_19_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_h_modules_19_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_78 = torch._C._nn.linear(
            layernorm_output_39,
            l_self_modules_h_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_39 = l_self_modules_h_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_19_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_116 = linear_78 * 0.5
        mul_117 = 0.79788456 * linear_78
        mul_118 = 0.044715 * linear_78
        mul_119 = mul_118 * linear_78
        mul_118 = linear_78 = None
        add_98 = 1 + mul_119
        mul_119 = None
        mul_120 = mul_117 * add_98
        mul_117 = add_98 = None
        tanh_19 = torch.tanh(mul_120)
        mul_120 = None
        add_99 = 1.0 + tanh_19
        tanh_19 = None
        hidden_states_20 = mul_116 * add_99
        mul_116 = add_99 = None
        intermediate_output_19 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_h_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_h_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_19_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_78 = torch.nn.functional.dropout(
            intermediate_output_19, p=0.0, training=False
        )
        intermediate_output_19 = None
        out_79 = out_77 + out_78
        out_77 = out_78 = None
        layernorm_output_40 = torch.nn.functional.layer_norm(
            out_79,
            (1024,),
            l_self_modules_h_modules_20_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_20_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_20_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_20_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_40 = torch._C._nn.linear(
            layernorm_output_40,
            l_self_modules_h_modules_20_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_20_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_40 = l_self_modules_h_modules_20_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_20_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_41 = fused_qkv_40.view(1, 9, 16, 3, 64)
        fused_qkv_40 = None
        getitem_86 = fused_qkv_41[(Ellipsis, 0, slice(None, None, None))]
        query_layer_40 = getitem_86.transpose(1, 2)
        getitem_86 = None
        getitem_87 = fused_qkv_41[(Ellipsis, 1, slice(None, None, None))]
        key_layer_40 = getitem_87.transpose(1, 2)
        getitem_87 = None
        getitem_88 = fused_qkv_41[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_41 = None
        value_layer_40 = getitem_88.transpose(1, 2)
        getitem_88 = None
        query_layer_41 = query_layer_40.reshape(16, -1, 64)
        query_layer_40 = None
        reshape_83 = key_layer_40.reshape(16, -1, 64)
        key_layer_41 = reshape_83.transpose(-1, -2)
        reshape_83 = None
        value_layer_41 = value_layer_40.reshape(16, -1, 64)
        attention_scores_20 = alibi_1.baddbmm(
            batch1=query_layer_41, batch2=key_layer_41, beta=1.0, alpha=0.125
        )
        query_layer_41 = key_layer_41 = None
        attn_weights_40 = attention_scores_20.view(1, 16, 9, -1)
        attention_scores_20 = None
        causal_mask_25 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_41 = attn_weights_40 + causal_mask_25
        attn_weights_40 = causal_mask_25 = None
        softmax_20 = torch.nn.functional.softmax(
            attn_weights_41, dim=-1, dtype=torch.float32
        )
        attn_weights_41 = None
        attention_probs_40 = softmax_20.to(torch.bfloat16)
        softmax_20 = None
        attention_probs_41 = torch.nn.functional.dropout(
            attention_probs_40, 0.0, False, False
        )
        attention_probs_40 = None
        attention_probs_reshaped_20 = attention_probs_41.view(16, 9, -1)
        attention_probs_41 = None
        context_layer_40 = torch.bmm(attention_probs_reshaped_20, value_layer_41)
        attention_probs_reshaped_20 = value_layer_41 = None
        x_40 = context_layer_40.view(1, 16, 9, 64)
        context_layer_40 = None
        x_41 = x_40.permute(0, 2, 1, 3)
        x_40 = None
        context_layer_41 = x_41.reshape(1, 9, 1024)
        x_41 = None
        output_tensor_20 = torch._C._nn.linear(
            context_layer_41,
            l_self_modules_h_modules_20_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_20_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_41 = l_self_modules_h_modules_20_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_20_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_80 = torch.nn.functional.dropout(output_tensor_20, p=0.0, training=False)
        output_tensor_20 = None
        out_81 = out_79 + out_80
        out_79 = out_80 = None
        layernorm_output_41 = torch.nn.functional.layer_norm(
            out_81,
            (1024,),
            l_self_modules_h_modules_20_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_20_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_20_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_h_modules_20_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_82 = torch._C._nn.linear(
            layernorm_output_41,
            l_self_modules_h_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_41 = l_self_modules_h_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_20_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_122 = linear_82 * 0.5
        mul_123 = 0.79788456 * linear_82
        mul_124 = 0.044715 * linear_82
        mul_125 = mul_124 * linear_82
        mul_124 = linear_82 = None
        add_103 = 1 + mul_125
        mul_125 = None
        mul_126 = mul_123 * add_103
        mul_123 = add_103 = None
        tanh_20 = torch.tanh(mul_126)
        mul_126 = None
        add_104 = 1.0 + tanh_20
        tanh_20 = None
        hidden_states_21 = mul_122 * add_104
        mul_122 = add_104 = None
        intermediate_output_20 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_h_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_21 = l_self_modules_h_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_20_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_82 = torch.nn.functional.dropout(
            intermediate_output_20, p=0.0, training=False
        )
        intermediate_output_20 = None
        out_83 = out_81 + out_82
        out_81 = out_82 = None
        layernorm_output_42 = torch.nn.functional.layer_norm(
            out_83,
            (1024,),
            l_self_modules_h_modules_21_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_21_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_21_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_21_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_42 = torch._C._nn.linear(
            layernorm_output_42,
            l_self_modules_h_modules_21_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_21_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_42 = l_self_modules_h_modules_21_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_21_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_43 = fused_qkv_42.view(1, 9, 16, 3, 64)
        fused_qkv_42 = None
        getitem_90 = fused_qkv_43[(Ellipsis, 0, slice(None, None, None))]
        query_layer_42 = getitem_90.transpose(1, 2)
        getitem_90 = None
        getitem_91 = fused_qkv_43[(Ellipsis, 1, slice(None, None, None))]
        key_layer_42 = getitem_91.transpose(1, 2)
        getitem_91 = None
        getitem_92 = fused_qkv_43[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_43 = None
        value_layer_42 = getitem_92.transpose(1, 2)
        getitem_92 = None
        query_layer_43 = query_layer_42.reshape(16, -1, 64)
        query_layer_42 = None
        reshape_87 = key_layer_42.reshape(16, -1, 64)
        key_layer_43 = reshape_87.transpose(-1, -2)
        reshape_87 = None
        value_layer_43 = value_layer_42.reshape(16, -1, 64)
        attention_scores_21 = alibi_1.baddbmm(
            batch1=query_layer_43, batch2=key_layer_43, beta=1.0, alpha=0.125
        )
        query_layer_43 = key_layer_43 = None
        attn_weights_42 = attention_scores_21.view(1, 16, 9, -1)
        attention_scores_21 = None
        causal_mask_26 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_43 = attn_weights_42 + causal_mask_26
        attn_weights_42 = causal_mask_26 = None
        softmax_21 = torch.nn.functional.softmax(
            attn_weights_43, dim=-1, dtype=torch.float32
        )
        attn_weights_43 = None
        attention_probs_42 = softmax_21.to(torch.bfloat16)
        softmax_21 = None
        attention_probs_43 = torch.nn.functional.dropout(
            attention_probs_42, 0.0, False, False
        )
        attention_probs_42 = None
        attention_probs_reshaped_21 = attention_probs_43.view(16, 9, -1)
        attention_probs_43 = None
        context_layer_42 = torch.bmm(attention_probs_reshaped_21, value_layer_43)
        attention_probs_reshaped_21 = value_layer_43 = None
        x_42 = context_layer_42.view(1, 16, 9, 64)
        context_layer_42 = None
        x_43 = x_42.permute(0, 2, 1, 3)
        x_42 = None
        context_layer_43 = x_43.reshape(1, 9, 1024)
        x_43 = None
        output_tensor_21 = torch._C._nn.linear(
            context_layer_43,
            l_self_modules_h_modules_21_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_21_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_43 = l_self_modules_h_modules_21_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_21_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_84 = torch.nn.functional.dropout(output_tensor_21, p=0.0, training=False)
        output_tensor_21 = None
        out_85 = out_83 + out_84
        out_83 = out_84 = None
        layernorm_output_43 = torch.nn.functional.layer_norm(
            out_85,
            (1024,),
            l_self_modules_h_modules_21_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_21_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_21_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_h_modules_21_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_86 = torch._C._nn.linear(
            layernorm_output_43,
            l_self_modules_h_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_43 = l_self_modules_h_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_21_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_128 = linear_86 * 0.5
        mul_129 = 0.79788456 * linear_86
        mul_130 = 0.044715 * linear_86
        mul_131 = mul_130 * linear_86
        mul_130 = linear_86 = None
        add_108 = 1 + mul_131
        mul_131 = None
        mul_132 = mul_129 * add_108
        mul_129 = add_108 = None
        tanh_21 = torch.tanh(mul_132)
        mul_132 = None
        add_109 = 1.0 + tanh_21
        tanh_21 = None
        hidden_states_22 = mul_128 * add_109
        mul_128 = add_109 = None
        intermediate_output_21 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_h_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_22 = l_self_modules_h_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_21_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_86 = torch.nn.functional.dropout(
            intermediate_output_21, p=0.0, training=False
        )
        intermediate_output_21 = None
        out_87 = out_85 + out_86
        out_85 = out_86 = None
        layernorm_output_44 = torch.nn.functional.layer_norm(
            out_87,
            (1024,),
            l_self_modules_h_modules_22_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_22_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_22_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_22_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_44 = torch._C._nn.linear(
            layernorm_output_44,
            l_self_modules_h_modules_22_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_22_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_44 = l_self_modules_h_modules_22_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_22_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_45 = fused_qkv_44.view(1, 9, 16, 3, 64)
        fused_qkv_44 = None
        getitem_94 = fused_qkv_45[(Ellipsis, 0, slice(None, None, None))]
        query_layer_44 = getitem_94.transpose(1, 2)
        getitem_94 = None
        getitem_95 = fused_qkv_45[(Ellipsis, 1, slice(None, None, None))]
        key_layer_44 = getitem_95.transpose(1, 2)
        getitem_95 = None
        getitem_96 = fused_qkv_45[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_45 = None
        value_layer_44 = getitem_96.transpose(1, 2)
        getitem_96 = None
        query_layer_45 = query_layer_44.reshape(16, -1, 64)
        query_layer_44 = None
        reshape_91 = key_layer_44.reshape(16, -1, 64)
        key_layer_45 = reshape_91.transpose(-1, -2)
        reshape_91 = None
        value_layer_45 = value_layer_44.reshape(16, -1, 64)
        attention_scores_22 = alibi_1.baddbmm(
            batch1=query_layer_45, batch2=key_layer_45, beta=1.0, alpha=0.125
        )
        query_layer_45 = key_layer_45 = None
        attn_weights_44 = attention_scores_22.view(1, 16, 9, -1)
        attention_scores_22 = None
        causal_mask_27 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        attn_weights_45 = attn_weights_44 + causal_mask_27
        attn_weights_44 = causal_mask_27 = None
        softmax_22 = torch.nn.functional.softmax(
            attn_weights_45, dim=-1, dtype=torch.float32
        )
        attn_weights_45 = None
        attention_probs_44 = softmax_22.to(torch.bfloat16)
        softmax_22 = None
        attention_probs_45 = torch.nn.functional.dropout(
            attention_probs_44, 0.0, False, False
        )
        attention_probs_44 = None
        attention_probs_reshaped_22 = attention_probs_45.view(16, 9, -1)
        attention_probs_45 = None
        context_layer_44 = torch.bmm(attention_probs_reshaped_22, value_layer_45)
        attention_probs_reshaped_22 = value_layer_45 = None
        x_44 = context_layer_44.view(1, 16, 9, 64)
        context_layer_44 = None
        x_45 = x_44.permute(0, 2, 1, 3)
        x_44 = None
        context_layer_45 = x_45.reshape(1, 9, 1024)
        x_45 = None
        output_tensor_22 = torch._C._nn.linear(
            context_layer_45,
            l_self_modules_h_modules_22_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_22_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_45 = l_self_modules_h_modules_22_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_22_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_88 = torch.nn.functional.dropout(output_tensor_22, p=0.0, training=False)
        output_tensor_22 = None
        out_89 = out_87 + out_88
        out_87 = out_88 = None
        layernorm_output_45 = torch.nn.functional.layer_norm(
            out_89,
            (1024,),
            l_self_modules_h_modules_22_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_22_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_22_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_h_modules_22_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_90 = torch._C._nn.linear(
            layernorm_output_45,
            l_self_modules_h_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_45 = l_self_modules_h_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_22_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_134 = linear_90 * 0.5
        mul_135 = 0.79788456 * linear_90
        mul_136 = 0.044715 * linear_90
        mul_137 = mul_136 * linear_90
        mul_136 = linear_90 = None
        add_113 = 1 + mul_137
        mul_137 = None
        mul_138 = mul_135 * add_113
        mul_135 = add_113 = None
        tanh_22 = torch.tanh(mul_138)
        mul_138 = None
        add_114 = 1.0 + tanh_22
        tanh_22 = None
        hidden_states_23 = mul_134 * add_114
        mul_134 = add_114 = None
        intermediate_output_22 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_h_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_23 = l_self_modules_h_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_22_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_90 = torch.nn.functional.dropout(
            intermediate_output_22, p=0.0, training=False
        )
        intermediate_output_22 = None
        out_91 = out_89 + out_90
        out_89 = out_90 = None
        layernorm_output_46 = torch.nn.functional.layer_norm(
            out_91,
            (1024,),
            l_self_modules_h_modules_23_modules_input_layernorm_parameters_weight_,
            l_self_modules_h_modules_23_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_23_modules_input_layernorm_parameters_weight_ = (
            l_self_modules_h_modules_23_modules_input_layernorm_parameters_bias_
        ) = None
        fused_qkv_46 = torch._C._nn.linear(
            layernorm_output_46,
            l_self_modules_h_modules_23_modules_self_attention_modules_query_key_value_parameters_weight_,
            l_self_modules_h_modules_23_modules_self_attention_modules_query_key_value_parameters_bias_,
        )
        layernorm_output_46 = l_self_modules_h_modules_23_modules_self_attention_modules_query_key_value_parameters_weight_ = l_self_modules_h_modules_23_modules_self_attention_modules_query_key_value_parameters_bias_ = (None)
        fused_qkv_47 = fused_qkv_46.view(1, 9, 16, 3, 64)
        fused_qkv_46 = None
        getitem_98 = fused_qkv_47[(Ellipsis, 0, slice(None, None, None))]
        query_layer_46 = getitem_98.transpose(1, 2)
        getitem_98 = None
        getitem_99 = fused_qkv_47[(Ellipsis, 1, slice(None, None, None))]
        key_layer_46 = getitem_99.transpose(1, 2)
        getitem_99 = None
        getitem_100 = fused_qkv_47[(Ellipsis, 2, slice(None, None, None))]
        fused_qkv_47 = None
        value_layer_46 = getitem_100.transpose(1, 2)
        getitem_100 = None
        query_layer_47 = query_layer_46.reshape(16, -1, 64)
        query_layer_46 = None
        reshape_95 = key_layer_46.reshape(16, -1, 64)
        key_layer_47 = reshape_95.transpose(-1, -2)
        reshape_95 = None
        value_layer_47 = value_layer_46.reshape(16, -1, 64)
        attention_scores_23 = alibi_1.baddbmm(
            batch1=query_layer_47, batch2=key_layer_47, beta=1.0, alpha=0.125
        )
        alibi_1 = query_layer_47 = key_layer_47 = None
        attn_weights_46 = attention_scores_23.view(1, 16, 9, -1)
        attention_scores_23 = None
        causal_mask_28 = causal_mask_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 9, None),
            )
        ]
        causal_mask_4 = None
        attn_weights_47 = attn_weights_46 + causal_mask_28
        attn_weights_46 = causal_mask_28 = None
        softmax_23 = torch.nn.functional.softmax(
            attn_weights_47, dim=-1, dtype=torch.float32
        )
        attn_weights_47 = None
        attention_probs_46 = softmax_23.to(torch.bfloat16)
        softmax_23 = None
        attention_probs_47 = torch.nn.functional.dropout(
            attention_probs_46, 0.0, False, False
        )
        attention_probs_46 = None
        attention_probs_reshaped_23 = attention_probs_47.view(16, 9, -1)
        attention_probs_47 = None
        context_layer_46 = torch.bmm(attention_probs_reshaped_23, value_layer_47)
        attention_probs_reshaped_23 = value_layer_47 = None
        x_46 = context_layer_46.view(1, 16, 9, 64)
        context_layer_46 = None
        x_47 = x_46.permute(0, 2, 1, 3)
        x_46 = None
        context_layer_47 = x_47.reshape(1, 9, 1024)
        x_47 = None
        output_tensor_23 = torch._C._nn.linear(
            context_layer_47,
            l_self_modules_h_modules_23_modules_self_attention_modules_dense_parameters_weight_,
            l_self_modules_h_modules_23_modules_self_attention_modules_dense_parameters_bias_,
        )
        context_layer_47 = l_self_modules_h_modules_23_modules_self_attention_modules_dense_parameters_weight_ = l_self_modules_h_modules_23_modules_self_attention_modules_dense_parameters_bias_ = (None)
        out_92 = torch.nn.functional.dropout(output_tensor_23, p=0.0, training=False)
        output_tensor_23 = None
        out_93 = out_91 + out_92
        out_91 = out_92 = None
        layernorm_output_47 = torch.nn.functional.layer_norm(
            out_93,
            (1024,),
            l_self_modules_h_modules_23_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_h_modules_23_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_h_modules_23_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_h_modules_23_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_94 = torch._C._nn.linear(
            layernorm_output_47,
            l_self_modules_h_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_weight_,
            l_self_modules_h_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_bias_,
        )
        layernorm_output_47 = l_self_modules_h_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_weight_ = l_self_modules_h_modules_23_modules_mlp_modules_dense_h_to_4h_parameters_bias_ = (None)
        mul_140 = linear_94 * 0.5
        mul_141 = 0.79788456 * linear_94
        mul_142 = 0.044715 * linear_94
        mul_143 = mul_142 * linear_94
        mul_142 = linear_94 = None
        add_118 = 1 + mul_143
        mul_143 = None
        mul_144 = mul_141 * add_118
        mul_141 = add_118 = None
        tanh_23 = torch.tanh(mul_144)
        mul_144 = None
        add_119 = 1.0 + tanh_23
        tanh_23 = None
        hidden_states_24 = mul_140 * add_119
        mul_140 = add_119 = None
        intermediate_output_23 = torch._C._nn.linear(
            hidden_states_24,
            l_self_modules_h_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_weight_,
            l_self_modules_h_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_bias_,
        )
        hidden_states_24 = l_self_modules_h_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_weight_ = l_self_modules_h_modules_23_modules_mlp_modules_dense_4h_to_h_parameters_bias_ = (None)
        out_94 = torch.nn.functional.dropout(
            intermediate_output_23, p=0.0, training=False
        )
        intermediate_output_23 = None
        out_95 = out_93 + out_94
        out_93 = out_94 = None
        hidden_states_25 = torch.nn.functional.layer_norm(
            out_95,
            (1024,),
            l_self_modules_ln_f_parameters_weight_,
            l_self_modules_ln_f_parameters_bias_,
            1e-05,
        )
        out_95 = (
            l_self_modules_ln_f_parameters_weight_
        ) = l_self_modules_ln_f_parameters_bias_ = None
        return (
            value_layer,
            key_layer,
            value_layer_2,
            key_layer_2,
            value_layer_4,
            key_layer_4,
            value_layer_6,
            key_layer_6,
            value_layer_8,
            key_layer_8,
            value_layer_10,
            key_layer_10,
            value_layer_12,
            key_layer_12,
            value_layer_14,
            key_layer_14,
            value_layer_16,
            key_layer_16,
            value_layer_18,
            key_layer_18,
            value_layer_20,
            key_layer_20,
            value_layer_22,
            key_layer_22,
            value_layer_24,
            key_layer_24,
            value_layer_26,
            key_layer_26,
            value_layer_28,
            key_layer_28,
            value_layer_30,
            key_layer_30,
            value_layer_32,
            key_layer_32,
            value_layer_34,
            key_layer_34,
            value_layer_36,
            key_layer_36,
            value_layer_38,
            key_layer_38,
            value_layer_40,
            key_layer_40,
            value_layer_42,
            key_layer_42,
            value_layer_44,
            key_layer_44,
            value_layer_46,
            key_layer_46,
            hidden_states_25,
        )
