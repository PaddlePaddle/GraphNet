import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_rel_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_embeddings_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_weight_ = (
            L_self_modules_embeddings_modules_LayerNorm_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = (
            L_self_modules_embeddings_modules_LayerNorm_parameters_bias_
        )
        l_self_modules_encoder_modules_rel_embeddings_parameters_weight_ = (
            L_self_modules_encoder_modules_rel_embeddings_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_norm_parameters_weight_ = (
            L_self_modules_encoder_modules_LayerNorm_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_norm_parameters_bias_ = (
            L_self_modules_encoder_modules_LayerNorm_parameters_bias_
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_proj_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_proj_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_proj_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_proj_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_LayerNorm_parameters_bias_
        inputs_embeds = torch.nn.functional.embedding(
            l_input_ids_,
            l_self_modules_embeddings_modules_word_embeddings_parameters_weight_,
            0,
            None,
            2.0,
            False,
            False,
        )
        l_input_ids_ = (
            l_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        ) = None
        position_embeddings = torch.zeros_like(inputs_embeds)
        position_embeddings = None
        embeddings = torch.nn.functional.layer_norm(
            inputs_embeds,
            (1024,),
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_,
            l_self_modules_embeddings_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        inputs_embeds = (
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_
        ) = l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = None
        mask = l_attention_mask_.unsqueeze(2)
        mask_1 = mask.to(torch.float32)
        mask = None
        embeddings_1 = embeddings * mask_1
        embeddings = mask_1 = None
        embeddings_2 = torch.nn.functional.dropout(embeddings_1, 0.1, False, False)
        embeddings_1 = None
        unsqueeze_1 = l_attention_mask_.unsqueeze(1)
        l_attention_mask_ = None
        extended_attention_mask = unsqueeze_1.unsqueeze(2)
        unsqueeze_1 = None
        squeeze = extended_attention_mask.squeeze(-2)
        unsqueeze_3 = squeeze.unsqueeze(-1)
        squeeze = None
        attention_mask = extended_attention_mask * unsqueeze_3
        extended_attention_mask = unsqueeze_3 = None
        q_ids = torch.arange(11, dtype=torch.int64, device=device(type="cuda", index=0))
        k_ids = torch.arange(11, dtype=torch.int64, device=device(type="cuda", index=0))
        getitem_1 = q_ids[(slice(None, None, None), None)]
        q_ids = None
        getitem_2 = k_ids[(None, slice(None, None, None))]
        k_ids = None
        rel_pos_ids = getitem_1 - getitem_2
        getitem_1 = getitem_2 = None
        sign = torch.sign(rel_pos_ids)
        lt = rel_pos_ids < 128
        gt = rel_pos_ids > -128
        and_ = lt & gt
        lt = gt = None
        tensor = torch.tensor(127)
        type_as = tensor.type_as(rel_pos_ids)
        tensor = None
        abs_1 = torch.abs(rel_pos_ids)
        abs_pos = torch.where(and_, type_as, abs_1)
        and_ = type_as = abs_1 = None
        truediv = abs_pos / 128
        log = torch.log(truediv)
        truediv = None
        tensor_1 = torch.tensor(3.9921875)
        log_1 = torch.log(tensor_1)
        tensor_1 = None
        truediv_1 = log / log_1
        log = log_1 = None
        mul_2 = truediv_1 * 127
        truediv_1 = None
        ceil = torch.ceil(mul_2)
        mul_2 = None
        log_pos = ceil + 128
        ceil = None
        le = abs_pos <= 128
        abs_pos = None
        type_as_1 = rel_pos_ids.type_as(log_pos)
        rel_pos_ids = None
        mul_3 = log_pos * sign
        log_pos = sign = None
        bucket_pos = torch.where(le, type_as_1, mul_3)
        le = type_as_1 = mul_3 = None
        rel_pos_ids_1 = bucket_pos.to(torch.int64)
        bucket_pos = None
        rel_pos_ids_2 = rel_pos_ids_1[(slice(None, 11, None), slice(None, None, None))]
        rel_pos_ids_1 = None
        rel_pos_ids_3 = rel_pos_ids_2.unsqueeze(0)
        rel_pos_ids_2 = None
        rel_embeddings = torch.nn.functional.layer_norm(
            l_self_modules_encoder_modules_rel_embeddings_parameters_weight_,
            (1024,),
            l_self_modules_encoder_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        l_self_modules_encoder_modules_rel_embeddings_parameters_weight_ = (
            l_self_modules_encoder_modules_layer_norm_parameters_weight_
        ) = l_self_modules_encoder_modules_layer_norm_parameters_bias_ = None
        linear = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x = linear.view((1, 11, 16, -1))
        linear = None
        permute = x.permute(0, 2, 1, 3)
        x = None
        contiguous = permute.contiguous()
        permute = None
        query_layer = contiguous.view(-1, 11, 64)
        contiguous = None
        linear_1 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_1 = linear_1.view((1, 11, 16, -1))
        linear_1 = None
        permute_1 = x_1.permute(0, 2, 1, 3)
        x_1 = None
        contiguous_1 = permute_1.contiguous()
        permute_1 = None
        key_layer = contiguous_1.view(-1, 11, 64)
        contiguous_1 = None
        linear_2 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_2 = linear_2.view((1, 11, 16, -1))
        linear_2 = None
        permute_2 = x_2.permute(0, 2, 1, 3)
        x_2 = None
        contiguous_2 = permute_2.contiguous()
        permute_2 = None
        value_layer = contiguous_2.view(-1, 11, 64)
        contiguous_2 = None
        tensor_2 = torch.tensor(64, dtype=torch.float32)
        mul_4 = tensor_2 * 3
        tensor_2 = None
        scale = torch.sqrt(mul_4)
        mul_4 = None
        transpose = key_layer.transpose(-1, -2)
        to_2 = scale.to(dtype=torch.float32)
        scale = None
        truediv_2 = transpose / to_2
        transpose = to_2 = None
        attention_scores = torch.bmm(query_layer, truediv_2)
        truediv_2 = None
        rel_embeddings_1 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos = rel_pos_ids_3.unsqueeze(1)
        relative_pos_1 = relative_pos.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos = None
        getitem_4 = rel_embeddings_1[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_1 = None
        rel_embeddings_2 = getitem_4.unsqueeze(0)
        getitem_4 = None
        linear_3 = torch._C._nn.linear(
            rel_embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_3 = linear_3.view((1, 512, 16, -1))
        linear_3 = None
        permute_3 = x_3.permute(0, 2, 1, 3)
        x_3 = None
        contiguous_3 = permute_3.contiguous()
        permute_3 = None
        view_7 = contiguous_3.view(-1, 512, 64)
        contiguous_3 = None
        pos_query_layer = view_7.repeat(1, 1, 1)
        view_7 = None
        linear_4 = torch._C._nn.linear(
            rel_embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_2 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_4 = linear_4.view((1, 512, 16, -1))
        linear_4 = None
        permute_4 = x_4.permute(0, 2, 1, 3)
        x_4 = None
        contiguous_4 = permute_4.contiguous()
        permute_4 = None
        view_9 = contiguous_4.view(-1, 512, 64)
        contiguous_4 = None
        pos_key_layer = view_9.repeat(1, 1, 1)
        view_9 = None
        tensor_3 = torch.tensor(64, dtype=torch.float32)
        mul_5 = tensor_3 * 3
        tensor_3 = None
        scale_1 = torch.sqrt(mul_5)
        mul_5 = None
        transpose_1 = pos_key_layer.transpose(-1, -2)
        pos_key_layer = None
        c2p_att = torch.bmm(query_layer, transpose_1)
        query_layer = transpose_1 = None
        add_1 = relative_pos_1 + 256
        c2p_pos = torch.clamp(add_1, 0, 511)
        add_1 = None
        squeeze_1 = c2p_pos.squeeze(0)
        c2p_pos = None
        expand = squeeze_1.expand([16, 11, 11])
        squeeze_1 = None
        c2p_att_1 = torch.gather(c2p_att, dim=-1, index=expand)
        c2p_att = expand = None
        to_4 = scale_1.to(dtype=torch.float32)
        scale_1 = None
        truediv_3 = c2p_att_1 / to_4
        c2p_att_1 = to_4 = None
        score = 0 + truediv_3
        truediv_3 = None
        tensor_4 = torch.tensor(64, dtype=torch.float32)
        mul_6 = tensor_4 * 3
        tensor_4 = None
        scale_2 = torch.sqrt(mul_6)
        mul_6 = None
        neg = -relative_pos_1
        relative_pos_1 = None
        add_3 = neg + 256
        neg = None
        p2c_pos = torch.clamp(add_3, 0, 511)
        add_3 = None
        transpose_2 = pos_query_layer.transpose(-1, -2)
        pos_query_layer = None
        p2c_att = torch.bmm(key_layer, transpose_2)
        key_layer = transpose_2 = None
        squeeze_2 = p2c_pos.squeeze(0)
        p2c_pos = None
        expand_1 = squeeze_2.expand([16, 11, 11])
        squeeze_2 = None
        gather_1 = torch.gather(p2c_att, dim=-1, index=expand_1)
        p2c_att = expand_1 = None
        p2c_att_1 = gather_1.transpose(-1, -2)
        gather_1 = None
        to_5 = scale_2.to(dtype=torch.float32)
        scale_2 = None
        truediv_4 = p2c_att_1 / to_5
        p2c_att_1 = to_5 = None
        score += truediv_4
        score_1 = score
        score = truediv_4 = None
        attention_scores_1 = attention_scores + score_1
        attention_scores = score_1 = None
        attention_scores_2 = attention_scores_1.view(-1, 16, 11, 11)
        attention_scores_1 = None
        attention_mask_1 = attention_mask.bool()
        invert = ~attention_mask_1
        attention_mask_1 = None
        attention_scores_3 = attention_scores_2.masked_fill(
            invert, -3.4028234663852886e38
        )
        attention_scores_2 = invert = None
        attention_probs = torch.nn.functional.softmax(attention_scores_3, dim=-1)
        attention_scores_3 = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.1, False, False
        )
        attention_probs = None
        view_11 = attention_probs_1.view(-1, 11, 11)
        attention_probs_1 = None
        context_layer = torch.bmm(view_11, value_layer)
        view_11 = value_layer = None
        view_12 = context_layer.view(-1, 16, 11, 64)
        context_layer = None
        permute_5 = view_12.permute(0, 2, 1, 3)
        view_12 = None
        context_layer_1 = permute_5.contiguous()
        permute_5 = None
        context_layer_2 = context_layer_1.view((1, 11, -1))
        context_layer_1 = None
        hidden_states = torch._C._nn.linear(
            context_layer_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.1, False, False)
        hidden_states = None
        add_5 = hidden_states_1 + embeddings_2
        hidden_states_1 = embeddings_2 = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            add_5,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_5 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_3 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_4 = torch._C._nn.gelu(hidden_states_3)
        hidden_states_3 = None
        hidden_states_5 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_4 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, 0.1, False, False
        )
        hidden_states_5 = None
        add_6 = hidden_states_6 + hidden_states_2
        hidden_states_6 = hidden_states_2 = None
        hidden_states_7 = torch.nn.functional.layer_norm(
            add_6,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_6 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_5 = linear_8.view((1, 11, 16, -1))
        linear_8 = None
        permute_6 = x_5.permute(0, 2, 1, 3)
        x_5 = None
        contiguous_6 = permute_6.contiguous()
        permute_6 = None
        query_layer_1 = contiguous_6.view(-1, 11, 64)
        contiguous_6 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_6 = linear_9.view((1, 11, 16, -1))
        linear_9 = None
        permute_7 = x_6.permute(0, 2, 1, 3)
        x_6 = None
        contiguous_7 = permute_7.contiguous()
        permute_7 = None
        key_layer_1 = contiguous_7.view(-1, 11, 64)
        contiguous_7 = None
        linear_10 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_7 = linear_10.view((1, 11, 16, -1))
        linear_10 = None
        permute_8 = x_7.permute(0, 2, 1, 3)
        x_7 = None
        contiguous_8 = permute_8.contiguous()
        permute_8 = None
        value_layer_1 = contiguous_8.view(-1, 11, 64)
        contiguous_8 = None
        tensor_5 = torch.tensor(64, dtype=torch.float32)
        mul_7 = tensor_5 * 3
        tensor_5 = None
        scale_3 = torch.sqrt(mul_7)
        mul_7 = None
        transpose_4 = key_layer_1.transpose(-1, -2)
        to_6 = scale_3.to(dtype=torch.float32)
        scale_3 = None
        truediv_5 = transpose_4 / to_6
        transpose_4 = to_6 = None
        attention_scores_4 = torch.bmm(query_layer_1, truediv_5)
        truediv_5 = None
        rel_embeddings_3 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_2 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_3 = relative_pos_2.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_2 = None
        getitem_5 = rel_embeddings_3[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_3 = None
        rel_embeddings_4 = getitem_5.unsqueeze(0)
        getitem_5 = None
        linear_11 = torch._C._nn.linear(
            rel_embeddings_4,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_8 = linear_11.view((1, 512, 16, -1))
        linear_11 = None
        permute_9 = x_8.permute(0, 2, 1, 3)
        x_8 = None
        contiguous_9 = permute_9.contiguous()
        permute_9 = None
        view_21 = contiguous_9.view(-1, 512, 64)
        contiguous_9 = None
        pos_query_layer_1 = view_21.repeat(1, 1, 1)
        view_21 = None
        linear_12 = torch._C._nn.linear(
            rel_embeddings_4,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_4 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_9 = linear_12.view((1, 512, 16, -1))
        linear_12 = None
        permute_10 = x_9.permute(0, 2, 1, 3)
        x_9 = None
        contiguous_10 = permute_10.contiguous()
        permute_10 = None
        view_23 = contiguous_10.view(-1, 512, 64)
        contiguous_10 = None
        pos_key_layer_1 = view_23.repeat(1, 1, 1)
        view_23 = None
        tensor_6 = torch.tensor(64, dtype=torch.float32)
        mul_8 = tensor_6 * 3
        tensor_6 = None
        scale_4 = torch.sqrt(mul_8)
        mul_8 = None
        transpose_5 = pos_key_layer_1.transpose(-1, -2)
        pos_key_layer_1 = None
        c2p_att_2 = torch.bmm(query_layer_1, transpose_5)
        query_layer_1 = transpose_5 = None
        add_7 = relative_pos_3 + 256
        c2p_pos_1 = torch.clamp(add_7, 0, 511)
        add_7 = None
        squeeze_3 = c2p_pos_1.squeeze(0)
        c2p_pos_1 = None
        expand_2 = squeeze_3.expand([16, 11, 11])
        squeeze_3 = None
        c2p_att_3 = torch.gather(c2p_att_2, dim=-1, index=expand_2)
        c2p_att_2 = expand_2 = None
        to_8 = scale_4.to(dtype=torch.float32)
        scale_4 = None
        truediv_6 = c2p_att_3 / to_8
        c2p_att_3 = to_8 = None
        score_2 = 0 + truediv_6
        truediv_6 = None
        tensor_7 = torch.tensor(64, dtype=torch.float32)
        mul_9 = tensor_7 * 3
        tensor_7 = None
        scale_5 = torch.sqrt(mul_9)
        mul_9 = None
        neg_1 = -relative_pos_3
        relative_pos_3 = None
        add_9 = neg_1 + 256
        neg_1 = None
        p2c_pos_1 = torch.clamp(add_9, 0, 511)
        add_9 = None
        transpose_6 = pos_query_layer_1.transpose(-1, -2)
        pos_query_layer_1 = None
        p2c_att_2 = torch.bmm(key_layer_1, transpose_6)
        key_layer_1 = transpose_6 = None
        squeeze_4 = p2c_pos_1.squeeze(0)
        p2c_pos_1 = None
        expand_3 = squeeze_4.expand([16, 11, 11])
        squeeze_4 = None
        gather_3 = torch.gather(p2c_att_2, dim=-1, index=expand_3)
        p2c_att_2 = expand_3 = None
        p2c_att_3 = gather_3.transpose(-1, -2)
        gather_3 = None
        to_9 = scale_5.to(dtype=torch.float32)
        scale_5 = None
        truediv_7 = p2c_att_3 / to_9
        p2c_att_3 = to_9 = None
        score_2 += truediv_7
        score_3 = score_2
        score_2 = truediv_7 = None
        attention_scores_5 = attention_scores_4 + score_3
        attention_scores_4 = score_3 = None
        attention_scores_6 = attention_scores_5.view(-1, 16, 11, 11)
        attention_scores_5 = None
        attention_mask_2 = attention_mask.bool()
        invert_1 = ~attention_mask_2
        attention_mask_2 = None
        attention_scores_7 = attention_scores_6.masked_fill(
            invert_1, -3.4028234663852886e38
        )
        attention_scores_6 = invert_1 = None
        attention_probs_2 = torch.nn.functional.softmax(attention_scores_7, dim=-1)
        attention_scores_7 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.1, False, False
        )
        attention_probs_2 = None
        view_25 = attention_probs_3.view(-1, 11, 11)
        attention_probs_3 = None
        context_layer_3 = torch.bmm(view_25, value_layer_1)
        view_25 = value_layer_1 = None
        view_26 = context_layer_3.view(-1, 16, 11, 64)
        context_layer_3 = None
        permute_11 = view_26.permute(0, 2, 1, 3)
        view_26 = None
        context_layer_4 = permute_11.contiguous()
        permute_11 = None
        context_layer_5 = context_layer_4.view((1, 11, -1))
        context_layer_4 = None
        hidden_states_8 = torch._C._nn.linear(
            context_layer_5,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_5 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, 0.1, False, False
        )
        hidden_states_8 = None
        add_11 = hidden_states_9 + hidden_states_7
        hidden_states_9 = hidden_states_7 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            add_11,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_11 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_11 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_12 = torch._C._nn.gelu(hidden_states_11)
        hidden_states_11 = None
        hidden_states_13 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_12 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_14 = torch.nn.functional.dropout(
            hidden_states_13, 0.1, False, False
        )
        hidden_states_13 = None
        add_12 = hidden_states_14 + hidden_states_10
        hidden_states_14 = hidden_states_10 = None
        hidden_states_15 = torch.nn.functional.layer_norm(
            add_12,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_12 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_16 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_10 = linear_16.view((1, 11, 16, -1))
        linear_16 = None
        permute_12 = x_10.permute(0, 2, 1, 3)
        x_10 = None
        contiguous_12 = permute_12.contiguous()
        permute_12 = None
        query_layer_2 = contiguous_12.view(-1, 11, 64)
        contiguous_12 = None
        linear_17 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_11 = linear_17.view((1, 11, 16, -1))
        linear_17 = None
        permute_13 = x_11.permute(0, 2, 1, 3)
        x_11 = None
        contiguous_13 = permute_13.contiguous()
        permute_13 = None
        key_layer_2 = contiguous_13.view(-1, 11, 64)
        contiguous_13 = None
        linear_18 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_12 = linear_18.view((1, 11, 16, -1))
        linear_18 = None
        permute_14 = x_12.permute(0, 2, 1, 3)
        x_12 = None
        contiguous_14 = permute_14.contiguous()
        permute_14 = None
        value_layer_2 = contiguous_14.view(-1, 11, 64)
        contiguous_14 = None
        tensor_8 = torch.tensor(64, dtype=torch.float32)
        mul_10 = tensor_8 * 3
        tensor_8 = None
        scale_6 = torch.sqrt(mul_10)
        mul_10 = None
        transpose_8 = key_layer_2.transpose(-1, -2)
        to_10 = scale_6.to(dtype=torch.float32)
        scale_6 = None
        truediv_8 = transpose_8 / to_10
        transpose_8 = to_10 = None
        attention_scores_8 = torch.bmm(query_layer_2, truediv_8)
        truediv_8 = None
        rel_embeddings_5 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_4 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_5 = relative_pos_4.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_4 = None
        getitem_6 = rel_embeddings_5[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_5 = None
        rel_embeddings_6 = getitem_6.unsqueeze(0)
        getitem_6 = None
        linear_19 = torch._C._nn.linear(
            rel_embeddings_6,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_13 = linear_19.view((1, 512, 16, -1))
        linear_19 = None
        permute_15 = x_13.permute(0, 2, 1, 3)
        x_13 = None
        contiguous_15 = permute_15.contiguous()
        permute_15 = None
        view_35 = contiguous_15.view(-1, 512, 64)
        contiguous_15 = None
        pos_query_layer_2 = view_35.repeat(1, 1, 1)
        view_35 = None
        linear_20 = torch._C._nn.linear(
            rel_embeddings_6,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_6 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_14 = linear_20.view((1, 512, 16, -1))
        linear_20 = None
        permute_16 = x_14.permute(0, 2, 1, 3)
        x_14 = None
        contiguous_16 = permute_16.contiguous()
        permute_16 = None
        view_37 = contiguous_16.view(-1, 512, 64)
        contiguous_16 = None
        pos_key_layer_2 = view_37.repeat(1, 1, 1)
        view_37 = None
        tensor_9 = torch.tensor(64, dtype=torch.float32)
        mul_11 = tensor_9 * 3
        tensor_9 = None
        scale_7 = torch.sqrt(mul_11)
        mul_11 = None
        transpose_9 = pos_key_layer_2.transpose(-1, -2)
        pos_key_layer_2 = None
        c2p_att_4 = torch.bmm(query_layer_2, transpose_9)
        query_layer_2 = transpose_9 = None
        add_13 = relative_pos_5 + 256
        c2p_pos_2 = torch.clamp(add_13, 0, 511)
        add_13 = None
        squeeze_5 = c2p_pos_2.squeeze(0)
        c2p_pos_2 = None
        expand_4 = squeeze_5.expand([16, 11, 11])
        squeeze_5 = None
        c2p_att_5 = torch.gather(c2p_att_4, dim=-1, index=expand_4)
        c2p_att_4 = expand_4 = None
        to_12 = scale_7.to(dtype=torch.float32)
        scale_7 = None
        truediv_9 = c2p_att_5 / to_12
        c2p_att_5 = to_12 = None
        score_4 = 0 + truediv_9
        truediv_9 = None
        tensor_10 = torch.tensor(64, dtype=torch.float32)
        mul_12 = tensor_10 * 3
        tensor_10 = None
        scale_8 = torch.sqrt(mul_12)
        mul_12 = None
        neg_2 = -relative_pos_5
        relative_pos_5 = None
        add_15 = neg_2 + 256
        neg_2 = None
        p2c_pos_2 = torch.clamp(add_15, 0, 511)
        add_15 = None
        transpose_10 = pos_query_layer_2.transpose(-1, -2)
        pos_query_layer_2 = None
        p2c_att_4 = torch.bmm(key_layer_2, transpose_10)
        key_layer_2 = transpose_10 = None
        squeeze_6 = p2c_pos_2.squeeze(0)
        p2c_pos_2 = None
        expand_5 = squeeze_6.expand([16, 11, 11])
        squeeze_6 = None
        gather_5 = torch.gather(p2c_att_4, dim=-1, index=expand_5)
        p2c_att_4 = expand_5 = None
        p2c_att_5 = gather_5.transpose(-1, -2)
        gather_5 = None
        to_13 = scale_8.to(dtype=torch.float32)
        scale_8 = None
        truediv_10 = p2c_att_5 / to_13
        p2c_att_5 = to_13 = None
        score_4 += truediv_10
        score_5 = score_4
        score_4 = truediv_10 = None
        attention_scores_9 = attention_scores_8 + score_5
        attention_scores_8 = score_5 = None
        attention_scores_10 = attention_scores_9.view(-1, 16, 11, 11)
        attention_scores_9 = None
        attention_mask_3 = attention_mask.bool()
        invert_2 = ~attention_mask_3
        attention_mask_3 = None
        attention_scores_11 = attention_scores_10.masked_fill(
            invert_2, -3.4028234663852886e38
        )
        attention_scores_10 = invert_2 = None
        attention_probs_4 = torch.nn.functional.softmax(attention_scores_11, dim=-1)
        attention_scores_11 = None
        attention_probs_5 = torch.nn.functional.dropout(
            attention_probs_4, 0.1, False, False
        )
        attention_probs_4 = None
        view_39 = attention_probs_5.view(-1, 11, 11)
        attention_probs_5 = None
        context_layer_6 = torch.bmm(view_39, value_layer_2)
        view_39 = value_layer_2 = None
        view_40 = context_layer_6.view(-1, 16, 11, 64)
        context_layer_6 = None
        permute_17 = view_40.permute(0, 2, 1, 3)
        view_40 = None
        context_layer_7 = permute_17.contiguous()
        permute_17 = None
        context_layer_8 = context_layer_7.view((1, 11, -1))
        context_layer_7 = None
        hidden_states_16 = torch._C._nn.linear(
            context_layer_8,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_8 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_17 = torch.nn.functional.dropout(
            hidden_states_16, 0.1, False, False
        )
        hidden_states_16 = None
        add_17 = hidden_states_17 + hidden_states_15
        hidden_states_17 = hidden_states_15 = None
        hidden_states_18 = torch.nn.functional.layer_norm(
            add_17,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_17 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_19 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_20 = torch._C._nn.gelu(hidden_states_19)
        hidden_states_19 = None
        hidden_states_21 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, 0.1, False, False
        )
        hidden_states_21 = None
        add_18 = hidden_states_22 + hidden_states_18
        hidden_states_22 = hidden_states_18 = None
        hidden_states_23 = torch.nn.functional.layer_norm(
            add_18,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_18 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_15 = linear_24.view((1, 11, 16, -1))
        linear_24 = None
        permute_18 = x_15.permute(0, 2, 1, 3)
        x_15 = None
        contiguous_18 = permute_18.contiguous()
        permute_18 = None
        query_layer_3 = contiguous_18.view(-1, 11, 64)
        contiguous_18 = None
        linear_25 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_16 = linear_25.view((1, 11, 16, -1))
        linear_25 = None
        permute_19 = x_16.permute(0, 2, 1, 3)
        x_16 = None
        contiguous_19 = permute_19.contiguous()
        permute_19 = None
        key_layer_3 = contiguous_19.view(-1, 11, 64)
        contiguous_19 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_17 = linear_26.view((1, 11, 16, -1))
        linear_26 = None
        permute_20 = x_17.permute(0, 2, 1, 3)
        x_17 = None
        contiguous_20 = permute_20.contiguous()
        permute_20 = None
        value_layer_3 = contiguous_20.view(-1, 11, 64)
        contiguous_20 = None
        tensor_11 = torch.tensor(64, dtype=torch.float32)
        mul_13 = tensor_11 * 3
        tensor_11 = None
        scale_9 = torch.sqrt(mul_13)
        mul_13 = None
        transpose_12 = key_layer_3.transpose(-1, -2)
        to_14 = scale_9.to(dtype=torch.float32)
        scale_9 = None
        truediv_11 = transpose_12 / to_14
        transpose_12 = to_14 = None
        attention_scores_12 = torch.bmm(query_layer_3, truediv_11)
        truediv_11 = None
        rel_embeddings_7 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_6 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_7 = relative_pos_6.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_6 = None
        getitem_7 = rel_embeddings_7[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_7 = None
        rel_embeddings_8 = getitem_7.unsqueeze(0)
        getitem_7 = None
        linear_27 = torch._C._nn.linear(
            rel_embeddings_8,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_18 = linear_27.view((1, 512, 16, -1))
        linear_27 = None
        permute_21 = x_18.permute(0, 2, 1, 3)
        x_18 = None
        contiguous_21 = permute_21.contiguous()
        permute_21 = None
        view_49 = contiguous_21.view(-1, 512, 64)
        contiguous_21 = None
        pos_query_layer_3 = view_49.repeat(1, 1, 1)
        view_49 = None
        linear_28 = torch._C._nn.linear(
            rel_embeddings_8,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_8 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_19 = linear_28.view((1, 512, 16, -1))
        linear_28 = None
        permute_22 = x_19.permute(0, 2, 1, 3)
        x_19 = None
        contiguous_22 = permute_22.contiguous()
        permute_22 = None
        view_51 = contiguous_22.view(-1, 512, 64)
        contiguous_22 = None
        pos_key_layer_3 = view_51.repeat(1, 1, 1)
        view_51 = None
        tensor_12 = torch.tensor(64, dtype=torch.float32)
        mul_14 = tensor_12 * 3
        tensor_12 = None
        scale_10 = torch.sqrt(mul_14)
        mul_14 = None
        transpose_13 = pos_key_layer_3.transpose(-1, -2)
        pos_key_layer_3 = None
        c2p_att_6 = torch.bmm(query_layer_3, transpose_13)
        query_layer_3 = transpose_13 = None
        add_19 = relative_pos_7 + 256
        c2p_pos_3 = torch.clamp(add_19, 0, 511)
        add_19 = None
        squeeze_7 = c2p_pos_3.squeeze(0)
        c2p_pos_3 = None
        expand_6 = squeeze_7.expand([16, 11, 11])
        squeeze_7 = None
        c2p_att_7 = torch.gather(c2p_att_6, dim=-1, index=expand_6)
        c2p_att_6 = expand_6 = None
        to_16 = scale_10.to(dtype=torch.float32)
        scale_10 = None
        truediv_12 = c2p_att_7 / to_16
        c2p_att_7 = to_16 = None
        score_6 = 0 + truediv_12
        truediv_12 = None
        tensor_13 = torch.tensor(64, dtype=torch.float32)
        mul_15 = tensor_13 * 3
        tensor_13 = None
        scale_11 = torch.sqrt(mul_15)
        mul_15 = None
        neg_3 = -relative_pos_7
        relative_pos_7 = None
        add_21 = neg_3 + 256
        neg_3 = None
        p2c_pos_3 = torch.clamp(add_21, 0, 511)
        add_21 = None
        transpose_14 = pos_query_layer_3.transpose(-1, -2)
        pos_query_layer_3 = None
        p2c_att_6 = torch.bmm(key_layer_3, transpose_14)
        key_layer_3 = transpose_14 = None
        squeeze_8 = p2c_pos_3.squeeze(0)
        p2c_pos_3 = None
        expand_7 = squeeze_8.expand([16, 11, 11])
        squeeze_8 = None
        gather_7 = torch.gather(p2c_att_6, dim=-1, index=expand_7)
        p2c_att_6 = expand_7 = None
        p2c_att_7 = gather_7.transpose(-1, -2)
        gather_7 = None
        to_17 = scale_11.to(dtype=torch.float32)
        scale_11 = None
        truediv_13 = p2c_att_7 / to_17
        p2c_att_7 = to_17 = None
        score_6 += truediv_13
        score_7 = score_6
        score_6 = truediv_13 = None
        attention_scores_13 = attention_scores_12 + score_7
        attention_scores_12 = score_7 = None
        attention_scores_14 = attention_scores_13.view(-1, 16, 11, 11)
        attention_scores_13 = None
        attention_mask_4 = attention_mask.bool()
        invert_3 = ~attention_mask_4
        attention_mask_4 = None
        attention_scores_15 = attention_scores_14.masked_fill(
            invert_3, -3.4028234663852886e38
        )
        attention_scores_14 = invert_3 = None
        attention_probs_6 = torch.nn.functional.softmax(attention_scores_15, dim=-1)
        attention_scores_15 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.1, False, False
        )
        attention_probs_6 = None
        view_53 = attention_probs_7.view(-1, 11, 11)
        attention_probs_7 = None
        context_layer_9 = torch.bmm(view_53, value_layer_3)
        view_53 = value_layer_3 = None
        view_54 = context_layer_9.view(-1, 16, 11, 64)
        context_layer_9 = None
        permute_23 = view_54.permute(0, 2, 1, 3)
        view_54 = None
        context_layer_10 = permute_23.contiguous()
        permute_23 = None
        context_layer_11 = context_layer_10.view((1, 11, -1))
        context_layer_10 = None
        hidden_states_24 = torch._C._nn.linear(
            context_layer_11,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, 0.1, False, False
        )
        hidden_states_24 = None
        add_23 = hidden_states_25 + hidden_states_23
        hidden_states_25 = hidden_states_23 = None
        hidden_states_26 = torch.nn.functional.layer_norm(
            add_23,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_23 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_27 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_28 = torch._C._nn.gelu(hidden_states_27)
        hidden_states_27 = None
        hidden_states_29 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_28 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_30 = torch.nn.functional.dropout(
            hidden_states_29, 0.1, False, False
        )
        hidden_states_29 = None
        add_24 = hidden_states_30 + hidden_states_26
        hidden_states_30 = hidden_states_26 = None
        hidden_states_31 = torch.nn.functional.layer_norm(
            add_24,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_24 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_32 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_20 = linear_32.view((1, 11, 16, -1))
        linear_32 = None
        permute_24 = x_20.permute(0, 2, 1, 3)
        x_20 = None
        contiguous_24 = permute_24.contiguous()
        permute_24 = None
        query_layer_4 = contiguous_24.view(-1, 11, 64)
        contiguous_24 = None
        linear_33 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_21 = linear_33.view((1, 11, 16, -1))
        linear_33 = None
        permute_25 = x_21.permute(0, 2, 1, 3)
        x_21 = None
        contiguous_25 = permute_25.contiguous()
        permute_25 = None
        key_layer_4 = contiguous_25.view(-1, 11, 64)
        contiguous_25 = None
        linear_34 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_22 = linear_34.view((1, 11, 16, -1))
        linear_34 = None
        permute_26 = x_22.permute(0, 2, 1, 3)
        x_22 = None
        contiguous_26 = permute_26.contiguous()
        permute_26 = None
        value_layer_4 = contiguous_26.view(-1, 11, 64)
        contiguous_26 = None
        tensor_14 = torch.tensor(64, dtype=torch.float32)
        mul_16 = tensor_14 * 3
        tensor_14 = None
        scale_12 = torch.sqrt(mul_16)
        mul_16 = None
        transpose_16 = key_layer_4.transpose(-1, -2)
        to_18 = scale_12.to(dtype=torch.float32)
        scale_12 = None
        truediv_14 = transpose_16 / to_18
        transpose_16 = to_18 = None
        attention_scores_16 = torch.bmm(query_layer_4, truediv_14)
        truediv_14 = None
        rel_embeddings_9 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_8 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_9 = relative_pos_8.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_8 = None
        getitem_8 = rel_embeddings_9[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_9 = None
        rel_embeddings_10 = getitem_8.unsqueeze(0)
        getitem_8 = None
        linear_35 = torch._C._nn.linear(
            rel_embeddings_10,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_23 = linear_35.view((1, 512, 16, -1))
        linear_35 = None
        permute_27 = x_23.permute(0, 2, 1, 3)
        x_23 = None
        contiguous_27 = permute_27.contiguous()
        permute_27 = None
        view_63 = contiguous_27.view(-1, 512, 64)
        contiguous_27 = None
        pos_query_layer_4 = view_63.repeat(1, 1, 1)
        view_63 = None
        linear_36 = torch._C._nn.linear(
            rel_embeddings_10,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_10 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_24 = linear_36.view((1, 512, 16, -1))
        linear_36 = None
        permute_28 = x_24.permute(0, 2, 1, 3)
        x_24 = None
        contiguous_28 = permute_28.contiguous()
        permute_28 = None
        view_65 = contiguous_28.view(-1, 512, 64)
        contiguous_28 = None
        pos_key_layer_4 = view_65.repeat(1, 1, 1)
        view_65 = None
        tensor_15 = torch.tensor(64, dtype=torch.float32)
        mul_17 = tensor_15 * 3
        tensor_15 = None
        scale_13 = torch.sqrt(mul_17)
        mul_17 = None
        transpose_17 = pos_key_layer_4.transpose(-1, -2)
        pos_key_layer_4 = None
        c2p_att_8 = torch.bmm(query_layer_4, transpose_17)
        query_layer_4 = transpose_17 = None
        add_25 = relative_pos_9 + 256
        c2p_pos_4 = torch.clamp(add_25, 0, 511)
        add_25 = None
        squeeze_9 = c2p_pos_4.squeeze(0)
        c2p_pos_4 = None
        expand_8 = squeeze_9.expand([16, 11, 11])
        squeeze_9 = None
        c2p_att_9 = torch.gather(c2p_att_8, dim=-1, index=expand_8)
        c2p_att_8 = expand_8 = None
        to_20 = scale_13.to(dtype=torch.float32)
        scale_13 = None
        truediv_15 = c2p_att_9 / to_20
        c2p_att_9 = to_20 = None
        score_8 = 0 + truediv_15
        truediv_15 = None
        tensor_16 = torch.tensor(64, dtype=torch.float32)
        mul_18 = tensor_16 * 3
        tensor_16 = None
        scale_14 = torch.sqrt(mul_18)
        mul_18 = None
        neg_4 = -relative_pos_9
        relative_pos_9 = None
        add_27 = neg_4 + 256
        neg_4 = None
        p2c_pos_4 = torch.clamp(add_27, 0, 511)
        add_27 = None
        transpose_18 = pos_query_layer_4.transpose(-1, -2)
        pos_query_layer_4 = None
        p2c_att_8 = torch.bmm(key_layer_4, transpose_18)
        key_layer_4 = transpose_18 = None
        squeeze_10 = p2c_pos_4.squeeze(0)
        p2c_pos_4 = None
        expand_9 = squeeze_10.expand([16, 11, 11])
        squeeze_10 = None
        gather_9 = torch.gather(p2c_att_8, dim=-1, index=expand_9)
        p2c_att_8 = expand_9 = None
        p2c_att_9 = gather_9.transpose(-1, -2)
        gather_9 = None
        to_21 = scale_14.to(dtype=torch.float32)
        scale_14 = None
        truediv_16 = p2c_att_9 / to_21
        p2c_att_9 = to_21 = None
        score_8 += truediv_16
        score_9 = score_8
        score_8 = truediv_16 = None
        attention_scores_17 = attention_scores_16 + score_9
        attention_scores_16 = score_9 = None
        attention_scores_18 = attention_scores_17.view(-1, 16, 11, 11)
        attention_scores_17 = None
        attention_mask_5 = attention_mask.bool()
        invert_4 = ~attention_mask_5
        attention_mask_5 = None
        attention_scores_19 = attention_scores_18.masked_fill(
            invert_4, -3.4028234663852886e38
        )
        attention_scores_18 = invert_4 = None
        attention_probs_8 = torch.nn.functional.softmax(attention_scores_19, dim=-1)
        attention_scores_19 = None
        attention_probs_9 = torch.nn.functional.dropout(
            attention_probs_8, 0.1, False, False
        )
        attention_probs_8 = None
        view_67 = attention_probs_9.view(-1, 11, 11)
        attention_probs_9 = None
        context_layer_12 = torch.bmm(view_67, value_layer_4)
        view_67 = value_layer_4 = None
        view_68 = context_layer_12.view(-1, 16, 11, 64)
        context_layer_12 = None
        permute_29 = view_68.permute(0, 2, 1, 3)
        view_68 = None
        context_layer_13 = permute_29.contiguous()
        permute_29 = None
        context_layer_14 = context_layer_13.view((1, 11, -1))
        context_layer_13 = None
        hidden_states_32 = torch._C._nn.linear(
            context_layer_14,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_14 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_33 = torch.nn.functional.dropout(
            hidden_states_32, 0.1, False, False
        )
        hidden_states_32 = None
        add_29 = hidden_states_33 + hidden_states_31
        hidden_states_33 = hidden_states_31 = None
        hidden_states_34 = torch.nn.functional.layer_norm(
            add_29,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_29 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_35 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_36 = torch._C._nn.gelu(hidden_states_35)
        hidden_states_35 = None
        hidden_states_37 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_36 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_38 = torch.nn.functional.dropout(
            hidden_states_37, 0.1, False, False
        )
        hidden_states_37 = None
        add_30 = hidden_states_38 + hidden_states_34
        hidden_states_38 = hidden_states_34 = None
        hidden_states_39 = torch.nn.functional.layer_norm(
            add_30,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_30 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_40 = torch._C._nn.linear(
            hidden_states_39,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_25 = linear_40.view((1, 11, 16, -1))
        linear_40 = None
        permute_30 = x_25.permute(0, 2, 1, 3)
        x_25 = None
        contiguous_30 = permute_30.contiguous()
        permute_30 = None
        query_layer_5 = contiguous_30.view(-1, 11, 64)
        contiguous_30 = None
        linear_41 = torch._C._nn.linear(
            hidden_states_39,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_26 = linear_41.view((1, 11, 16, -1))
        linear_41 = None
        permute_31 = x_26.permute(0, 2, 1, 3)
        x_26 = None
        contiguous_31 = permute_31.contiguous()
        permute_31 = None
        key_layer_5 = contiguous_31.view(-1, 11, 64)
        contiguous_31 = None
        linear_42 = torch._C._nn.linear(
            hidden_states_39,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_27 = linear_42.view((1, 11, 16, -1))
        linear_42 = None
        permute_32 = x_27.permute(0, 2, 1, 3)
        x_27 = None
        contiguous_32 = permute_32.contiguous()
        permute_32 = None
        value_layer_5 = contiguous_32.view(-1, 11, 64)
        contiguous_32 = None
        tensor_17 = torch.tensor(64, dtype=torch.float32)
        mul_19 = tensor_17 * 3
        tensor_17 = None
        scale_15 = torch.sqrt(mul_19)
        mul_19 = None
        transpose_20 = key_layer_5.transpose(-1, -2)
        to_22 = scale_15.to(dtype=torch.float32)
        scale_15 = None
        truediv_17 = transpose_20 / to_22
        transpose_20 = to_22 = None
        attention_scores_20 = torch.bmm(query_layer_5, truediv_17)
        truediv_17 = None
        rel_embeddings_11 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_10 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_11 = relative_pos_10.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_10 = None
        getitem_9 = rel_embeddings_11[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_11 = None
        rel_embeddings_12 = getitem_9.unsqueeze(0)
        getitem_9 = None
        linear_43 = torch._C._nn.linear(
            rel_embeddings_12,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_28 = linear_43.view((1, 512, 16, -1))
        linear_43 = None
        permute_33 = x_28.permute(0, 2, 1, 3)
        x_28 = None
        contiguous_33 = permute_33.contiguous()
        permute_33 = None
        view_77 = contiguous_33.view(-1, 512, 64)
        contiguous_33 = None
        pos_query_layer_5 = view_77.repeat(1, 1, 1)
        view_77 = None
        linear_44 = torch._C._nn.linear(
            rel_embeddings_12,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_12 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_29 = linear_44.view((1, 512, 16, -1))
        linear_44 = None
        permute_34 = x_29.permute(0, 2, 1, 3)
        x_29 = None
        contiguous_34 = permute_34.contiguous()
        permute_34 = None
        view_79 = contiguous_34.view(-1, 512, 64)
        contiguous_34 = None
        pos_key_layer_5 = view_79.repeat(1, 1, 1)
        view_79 = None
        tensor_18 = torch.tensor(64, dtype=torch.float32)
        mul_20 = tensor_18 * 3
        tensor_18 = None
        scale_16 = torch.sqrt(mul_20)
        mul_20 = None
        transpose_21 = pos_key_layer_5.transpose(-1, -2)
        pos_key_layer_5 = None
        c2p_att_10 = torch.bmm(query_layer_5, transpose_21)
        query_layer_5 = transpose_21 = None
        add_31 = relative_pos_11 + 256
        c2p_pos_5 = torch.clamp(add_31, 0, 511)
        add_31 = None
        squeeze_11 = c2p_pos_5.squeeze(0)
        c2p_pos_5 = None
        expand_10 = squeeze_11.expand([16, 11, 11])
        squeeze_11 = None
        c2p_att_11 = torch.gather(c2p_att_10, dim=-1, index=expand_10)
        c2p_att_10 = expand_10 = None
        to_24 = scale_16.to(dtype=torch.float32)
        scale_16 = None
        truediv_18 = c2p_att_11 / to_24
        c2p_att_11 = to_24 = None
        score_10 = 0 + truediv_18
        truediv_18 = None
        tensor_19 = torch.tensor(64, dtype=torch.float32)
        mul_21 = tensor_19 * 3
        tensor_19 = None
        scale_17 = torch.sqrt(mul_21)
        mul_21 = None
        neg_5 = -relative_pos_11
        relative_pos_11 = None
        add_33 = neg_5 + 256
        neg_5 = None
        p2c_pos_5 = torch.clamp(add_33, 0, 511)
        add_33 = None
        transpose_22 = pos_query_layer_5.transpose(-1, -2)
        pos_query_layer_5 = None
        p2c_att_10 = torch.bmm(key_layer_5, transpose_22)
        key_layer_5 = transpose_22 = None
        squeeze_12 = p2c_pos_5.squeeze(0)
        p2c_pos_5 = None
        expand_11 = squeeze_12.expand([16, 11, 11])
        squeeze_12 = None
        gather_11 = torch.gather(p2c_att_10, dim=-1, index=expand_11)
        p2c_att_10 = expand_11 = None
        p2c_att_11 = gather_11.transpose(-1, -2)
        gather_11 = None
        to_25 = scale_17.to(dtype=torch.float32)
        scale_17 = None
        truediv_19 = p2c_att_11 / to_25
        p2c_att_11 = to_25 = None
        score_10 += truediv_19
        score_11 = score_10
        score_10 = truediv_19 = None
        attention_scores_21 = attention_scores_20 + score_11
        attention_scores_20 = score_11 = None
        attention_scores_22 = attention_scores_21.view(-1, 16, 11, 11)
        attention_scores_21 = None
        attention_mask_6 = attention_mask.bool()
        invert_5 = ~attention_mask_6
        attention_mask_6 = None
        attention_scores_23 = attention_scores_22.masked_fill(
            invert_5, -3.4028234663852886e38
        )
        attention_scores_22 = invert_5 = None
        attention_probs_10 = torch.nn.functional.softmax(attention_scores_23, dim=-1)
        attention_scores_23 = None
        attention_probs_11 = torch.nn.functional.dropout(
            attention_probs_10, 0.1, False, False
        )
        attention_probs_10 = None
        view_81 = attention_probs_11.view(-1, 11, 11)
        attention_probs_11 = None
        context_layer_15 = torch.bmm(view_81, value_layer_5)
        view_81 = value_layer_5 = None
        view_82 = context_layer_15.view(-1, 16, 11, 64)
        context_layer_15 = None
        permute_35 = view_82.permute(0, 2, 1, 3)
        view_82 = None
        context_layer_16 = permute_35.contiguous()
        permute_35 = None
        context_layer_17 = context_layer_16.view((1, 11, -1))
        context_layer_16 = None
        hidden_states_40 = torch._C._nn.linear(
            context_layer_17,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_17 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_41 = torch.nn.functional.dropout(
            hidden_states_40, 0.1, False, False
        )
        hidden_states_40 = None
        add_35 = hidden_states_41 + hidden_states_39
        hidden_states_41 = hidden_states_39 = None
        hidden_states_42 = torch.nn.functional.layer_norm(
            add_35,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_35 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_43 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_44 = torch._C._nn.gelu(hidden_states_43)
        hidden_states_43 = None
        hidden_states_45 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_44 = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_46 = torch.nn.functional.dropout(
            hidden_states_45, 0.1, False, False
        )
        hidden_states_45 = None
        add_36 = hidden_states_46 + hidden_states_42
        hidden_states_46 = hidden_states_42 = None
        hidden_states_47 = torch.nn.functional.layer_norm(
            add_36,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_36 = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_30 = linear_48.view((1, 11, 16, -1))
        linear_48 = None
        permute_36 = x_30.permute(0, 2, 1, 3)
        x_30 = None
        contiguous_36 = permute_36.contiguous()
        permute_36 = None
        query_layer_6 = contiguous_36.view(-1, 11, 64)
        contiguous_36 = None
        linear_49 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_31 = linear_49.view((1, 11, 16, -1))
        linear_49 = None
        permute_37 = x_31.permute(0, 2, 1, 3)
        x_31 = None
        contiguous_37 = permute_37.contiguous()
        permute_37 = None
        key_layer_6 = contiguous_37.view(-1, 11, 64)
        contiguous_37 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_32 = linear_50.view((1, 11, 16, -1))
        linear_50 = None
        permute_38 = x_32.permute(0, 2, 1, 3)
        x_32 = None
        contiguous_38 = permute_38.contiguous()
        permute_38 = None
        value_layer_6 = contiguous_38.view(-1, 11, 64)
        contiguous_38 = None
        tensor_20 = torch.tensor(64, dtype=torch.float32)
        mul_22 = tensor_20 * 3
        tensor_20 = None
        scale_18 = torch.sqrt(mul_22)
        mul_22 = None
        transpose_24 = key_layer_6.transpose(-1, -2)
        to_26 = scale_18.to(dtype=torch.float32)
        scale_18 = None
        truediv_20 = transpose_24 / to_26
        transpose_24 = to_26 = None
        attention_scores_24 = torch.bmm(query_layer_6, truediv_20)
        truediv_20 = None
        rel_embeddings_13 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_12 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_13 = relative_pos_12.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_12 = None
        getitem_10 = rel_embeddings_13[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_13 = None
        rel_embeddings_14 = getitem_10.unsqueeze(0)
        getitem_10 = None
        linear_51 = torch._C._nn.linear(
            rel_embeddings_14,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_33 = linear_51.view((1, 512, 16, -1))
        linear_51 = None
        permute_39 = x_33.permute(0, 2, 1, 3)
        x_33 = None
        contiguous_39 = permute_39.contiguous()
        permute_39 = None
        view_91 = contiguous_39.view(-1, 512, 64)
        contiguous_39 = None
        pos_query_layer_6 = view_91.repeat(1, 1, 1)
        view_91 = None
        linear_52 = torch._C._nn.linear(
            rel_embeddings_14,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_14 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_34 = linear_52.view((1, 512, 16, -1))
        linear_52 = None
        permute_40 = x_34.permute(0, 2, 1, 3)
        x_34 = None
        contiguous_40 = permute_40.contiguous()
        permute_40 = None
        view_93 = contiguous_40.view(-1, 512, 64)
        contiguous_40 = None
        pos_key_layer_6 = view_93.repeat(1, 1, 1)
        view_93 = None
        tensor_21 = torch.tensor(64, dtype=torch.float32)
        mul_23 = tensor_21 * 3
        tensor_21 = None
        scale_19 = torch.sqrt(mul_23)
        mul_23 = None
        transpose_25 = pos_key_layer_6.transpose(-1, -2)
        pos_key_layer_6 = None
        c2p_att_12 = torch.bmm(query_layer_6, transpose_25)
        query_layer_6 = transpose_25 = None
        add_37 = relative_pos_13 + 256
        c2p_pos_6 = torch.clamp(add_37, 0, 511)
        add_37 = None
        squeeze_13 = c2p_pos_6.squeeze(0)
        c2p_pos_6 = None
        expand_12 = squeeze_13.expand([16, 11, 11])
        squeeze_13 = None
        c2p_att_13 = torch.gather(c2p_att_12, dim=-1, index=expand_12)
        c2p_att_12 = expand_12 = None
        to_28 = scale_19.to(dtype=torch.float32)
        scale_19 = None
        truediv_21 = c2p_att_13 / to_28
        c2p_att_13 = to_28 = None
        score_12 = 0 + truediv_21
        truediv_21 = None
        tensor_22 = torch.tensor(64, dtype=torch.float32)
        mul_24 = tensor_22 * 3
        tensor_22 = None
        scale_20 = torch.sqrt(mul_24)
        mul_24 = None
        neg_6 = -relative_pos_13
        relative_pos_13 = None
        add_39 = neg_6 + 256
        neg_6 = None
        p2c_pos_6 = torch.clamp(add_39, 0, 511)
        add_39 = None
        transpose_26 = pos_query_layer_6.transpose(-1, -2)
        pos_query_layer_6 = None
        p2c_att_12 = torch.bmm(key_layer_6, transpose_26)
        key_layer_6 = transpose_26 = None
        squeeze_14 = p2c_pos_6.squeeze(0)
        p2c_pos_6 = None
        expand_13 = squeeze_14.expand([16, 11, 11])
        squeeze_14 = None
        gather_13 = torch.gather(p2c_att_12, dim=-1, index=expand_13)
        p2c_att_12 = expand_13 = None
        p2c_att_13 = gather_13.transpose(-1, -2)
        gather_13 = None
        to_29 = scale_20.to(dtype=torch.float32)
        scale_20 = None
        truediv_22 = p2c_att_13 / to_29
        p2c_att_13 = to_29 = None
        score_12 += truediv_22
        score_13 = score_12
        score_12 = truediv_22 = None
        attention_scores_25 = attention_scores_24 + score_13
        attention_scores_24 = score_13 = None
        attention_scores_26 = attention_scores_25.view(-1, 16, 11, 11)
        attention_scores_25 = None
        attention_mask_7 = attention_mask.bool()
        invert_6 = ~attention_mask_7
        attention_mask_7 = None
        attention_scores_27 = attention_scores_26.masked_fill(
            invert_6, -3.4028234663852886e38
        )
        attention_scores_26 = invert_6 = None
        attention_probs_12 = torch.nn.functional.softmax(attention_scores_27, dim=-1)
        attention_scores_27 = None
        attention_probs_13 = torch.nn.functional.dropout(
            attention_probs_12, 0.1, False, False
        )
        attention_probs_12 = None
        view_95 = attention_probs_13.view(-1, 11, 11)
        attention_probs_13 = None
        context_layer_18 = torch.bmm(view_95, value_layer_6)
        view_95 = value_layer_6 = None
        view_96 = context_layer_18.view(-1, 16, 11, 64)
        context_layer_18 = None
        permute_41 = view_96.permute(0, 2, 1, 3)
        view_96 = None
        context_layer_19 = permute_41.contiguous()
        permute_41 = None
        context_layer_20 = context_layer_19.view((1, 11, -1))
        context_layer_19 = None
        hidden_states_48 = torch._C._nn.linear(
            context_layer_20,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_20 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_49 = torch.nn.functional.dropout(
            hidden_states_48, 0.1, False, False
        )
        hidden_states_48 = None
        add_41 = hidden_states_49 + hidden_states_47
        hidden_states_49 = hidden_states_47 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            add_41,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_41 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_51 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_52 = torch._C._nn.gelu(hidden_states_51)
        hidden_states_51 = None
        hidden_states_53 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_52 = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_54 = torch.nn.functional.dropout(
            hidden_states_53, 0.1, False, False
        )
        hidden_states_53 = None
        add_42 = hidden_states_54 + hidden_states_50
        hidden_states_54 = hidden_states_50 = None
        hidden_states_55 = torch.nn.functional.layer_norm(
            add_42,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_42 = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_56 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_35 = linear_56.view((1, 11, 16, -1))
        linear_56 = None
        permute_42 = x_35.permute(0, 2, 1, 3)
        x_35 = None
        contiguous_42 = permute_42.contiguous()
        permute_42 = None
        query_layer_7 = contiguous_42.view(-1, 11, 64)
        contiguous_42 = None
        linear_57 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_36 = linear_57.view((1, 11, 16, -1))
        linear_57 = None
        permute_43 = x_36.permute(0, 2, 1, 3)
        x_36 = None
        contiguous_43 = permute_43.contiguous()
        permute_43 = None
        key_layer_7 = contiguous_43.view(-1, 11, 64)
        contiguous_43 = None
        linear_58 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_37 = linear_58.view((1, 11, 16, -1))
        linear_58 = None
        permute_44 = x_37.permute(0, 2, 1, 3)
        x_37 = None
        contiguous_44 = permute_44.contiguous()
        permute_44 = None
        value_layer_7 = contiguous_44.view(-1, 11, 64)
        contiguous_44 = None
        tensor_23 = torch.tensor(64, dtype=torch.float32)
        mul_25 = tensor_23 * 3
        tensor_23 = None
        scale_21 = torch.sqrt(mul_25)
        mul_25 = None
        transpose_28 = key_layer_7.transpose(-1, -2)
        to_30 = scale_21.to(dtype=torch.float32)
        scale_21 = None
        truediv_23 = transpose_28 / to_30
        transpose_28 = to_30 = None
        attention_scores_28 = torch.bmm(query_layer_7, truediv_23)
        truediv_23 = None
        rel_embeddings_15 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_14 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_15 = relative_pos_14.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_14 = None
        getitem_11 = rel_embeddings_15[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_15 = None
        rel_embeddings_16 = getitem_11.unsqueeze(0)
        getitem_11 = None
        linear_59 = torch._C._nn.linear(
            rel_embeddings_16,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_38 = linear_59.view((1, 512, 16, -1))
        linear_59 = None
        permute_45 = x_38.permute(0, 2, 1, 3)
        x_38 = None
        contiguous_45 = permute_45.contiguous()
        permute_45 = None
        view_105 = contiguous_45.view(-1, 512, 64)
        contiguous_45 = None
        pos_query_layer_7 = view_105.repeat(1, 1, 1)
        view_105 = None
        linear_60 = torch._C._nn.linear(
            rel_embeddings_16,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_16 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_39 = linear_60.view((1, 512, 16, -1))
        linear_60 = None
        permute_46 = x_39.permute(0, 2, 1, 3)
        x_39 = None
        contiguous_46 = permute_46.contiguous()
        permute_46 = None
        view_107 = contiguous_46.view(-1, 512, 64)
        contiguous_46 = None
        pos_key_layer_7 = view_107.repeat(1, 1, 1)
        view_107 = None
        tensor_24 = torch.tensor(64, dtype=torch.float32)
        mul_26 = tensor_24 * 3
        tensor_24 = None
        scale_22 = torch.sqrt(mul_26)
        mul_26 = None
        transpose_29 = pos_key_layer_7.transpose(-1, -2)
        pos_key_layer_7 = None
        c2p_att_14 = torch.bmm(query_layer_7, transpose_29)
        query_layer_7 = transpose_29 = None
        add_43 = relative_pos_15 + 256
        c2p_pos_7 = torch.clamp(add_43, 0, 511)
        add_43 = None
        squeeze_15 = c2p_pos_7.squeeze(0)
        c2p_pos_7 = None
        expand_14 = squeeze_15.expand([16, 11, 11])
        squeeze_15 = None
        c2p_att_15 = torch.gather(c2p_att_14, dim=-1, index=expand_14)
        c2p_att_14 = expand_14 = None
        to_32 = scale_22.to(dtype=torch.float32)
        scale_22 = None
        truediv_24 = c2p_att_15 / to_32
        c2p_att_15 = to_32 = None
        score_14 = 0 + truediv_24
        truediv_24 = None
        tensor_25 = torch.tensor(64, dtype=torch.float32)
        mul_27 = tensor_25 * 3
        tensor_25 = None
        scale_23 = torch.sqrt(mul_27)
        mul_27 = None
        neg_7 = -relative_pos_15
        relative_pos_15 = None
        add_45 = neg_7 + 256
        neg_7 = None
        p2c_pos_7 = torch.clamp(add_45, 0, 511)
        add_45 = None
        transpose_30 = pos_query_layer_7.transpose(-1, -2)
        pos_query_layer_7 = None
        p2c_att_14 = torch.bmm(key_layer_7, transpose_30)
        key_layer_7 = transpose_30 = None
        squeeze_16 = p2c_pos_7.squeeze(0)
        p2c_pos_7 = None
        expand_15 = squeeze_16.expand([16, 11, 11])
        squeeze_16 = None
        gather_15 = torch.gather(p2c_att_14, dim=-1, index=expand_15)
        p2c_att_14 = expand_15 = None
        p2c_att_15 = gather_15.transpose(-1, -2)
        gather_15 = None
        to_33 = scale_23.to(dtype=torch.float32)
        scale_23 = None
        truediv_25 = p2c_att_15 / to_33
        p2c_att_15 = to_33 = None
        score_14 += truediv_25
        score_15 = score_14
        score_14 = truediv_25 = None
        attention_scores_29 = attention_scores_28 + score_15
        attention_scores_28 = score_15 = None
        attention_scores_30 = attention_scores_29.view(-1, 16, 11, 11)
        attention_scores_29 = None
        attention_mask_8 = attention_mask.bool()
        invert_7 = ~attention_mask_8
        attention_mask_8 = None
        attention_scores_31 = attention_scores_30.masked_fill(
            invert_7, -3.4028234663852886e38
        )
        attention_scores_30 = invert_7 = None
        attention_probs_14 = torch.nn.functional.softmax(attention_scores_31, dim=-1)
        attention_scores_31 = None
        attention_probs_15 = torch.nn.functional.dropout(
            attention_probs_14, 0.1, False, False
        )
        attention_probs_14 = None
        view_109 = attention_probs_15.view(-1, 11, 11)
        attention_probs_15 = None
        context_layer_21 = torch.bmm(view_109, value_layer_7)
        view_109 = value_layer_7 = None
        view_110 = context_layer_21.view(-1, 16, 11, 64)
        context_layer_21 = None
        permute_47 = view_110.permute(0, 2, 1, 3)
        view_110 = None
        context_layer_22 = permute_47.contiguous()
        permute_47 = None
        context_layer_23 = context_layer_22.view((1, 11, -1))
        context_layer_22 = None
        hidden_states_56 = torch._C._nn.linear(
            context_layer_23,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_23 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_57 = torch.nn.functional.dropout(
            hidden_states_56, 0.1, False, False
        )
        hidden_states_56 = None
        add_47 = hidden_states_57 + hidden_states_55
        hidden_states_57 = hidden_states_55 = None
        hidden_states_58 = torch.nn.functional.layer_norm(
            add_47,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_47 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_59 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_60 = torch._C._nn.gelu(hidden_states_59)
        hidden_states_59 = None
        hidden_states_61 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_60 = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_62 = torch.nn.functional.dropout(
            hidden_states_61, 0.1, False, False
        )
        hidden_states_61 = None
        add_48 = hidden_states_62 + hidden_states_58
        hidden_states_62 = hidden_states_58 = None
        hidden_states_63 = torch.nn.functional.layer_norm(
            add_48,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_48 = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_64 = torch._C._nn.linear(
            hidden_states_63,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_40 = linear_64.view((1, 11, 16, -1))
        linear_64 = None
        permute_48 = x_40.permute(0, 2, 1, 3)
        x_40 = None
        contiguous_48 = permute_48.contiguous()
        permute_48 = None
        query_layer_8 = contiguous_48.view(-1, 11, 64)
        contiguous_48 = None
        linear_65 = torch._C._nn.linear(
            hidden_states_63,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_41 = linear_65.view((1, 11, 16, -1))
        linear_65 = None
        permute_49 = x_41.permute(0, 2, 1, 3)
        x_41 = None
        contiguous_49 = permute_49.contiguous()
        permute_49 = None
        key_layer_8 = contiguous_49.view(-1, 11, 64)
        contiguous_49 = None
        linear_66 = torch._C._nn.linear(
            hidden_states_63,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_42 = linear_66.view((1, 11, 16, -1))
        linear_66 = None
        permute_50 = x_42.permute(0, 2, 1, 3)
        x_42 = None
        contiguous_50 = permute_50.contiguous()
        permute_50 = None
        value_layer_8 = contiguous_50.view(-1, 11, 64)
        contiguous_50 = None
        tensor_26 = torch.tensor(64, dtype=torch.float32)
        mul_28 = tensor_26 * 3
        tensor_26 = None
        scale_24 = torch.sqrt(mul_28)
        mul_28 = None
        transpose_32 = key_layer_8.transpose(-1, -2)
        to_34 = scale_24.to(dtype=torch.float32)
        scale_24 = None
        truediv_26 = transpose_32 / to_34
        transpose_32 = to_34 = None
        attention_scores_32 = torch.bmm(query_layer_8, truediv_26)
        truediv_26 = None
        rel_embeddings_17 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_16 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_17 = relative_pos_16.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_16 = None
        getitem_12 = rel_embeddings_17[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_17 = None
        rel_embeddings_18 = getitem_12.unsqueeze(0)
        getitem_12 = None
        linear_67 = torch._C._nn.linear(
            rel_embeddings_18,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_43 = linear_67.view((1, 512, 16, -1))
        linear_67 = None
        permute_51 = x_43.permute(0, 2, 1, 3)
        x_43 = None
        contiguous_51 = permute_51.contiguous()
        permute_51 = None
        view_119 = contiguous_51.view(-1, 512, 64)
        contiguous_51 = None
        pos_query_layer_8 = view_119.repeat(1, 1, 1)
        view_119 = None
        linear_68 = torch._C._nn.linear(
            rel_embeddings_18,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_18 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_44 = linear_68.view((1, 512, 16, -1))
        linear_68 = None
        permute_52 = x_44.permute(0, 2, 1, 3)
        x_44 = None
        contiguous_52 = permute_52.contiguous()
        permute_52 = None
        view_121 = contiguous_52.view(-1, 512, 64)
        contiguous_52 = None
        pos_key_layer_8 = view_121.repeat(1, 1, 1)
        view_121 = None
        tensor_27 = torch.tensor(64, dtype=torch.float32)
        mul_29 = tensor_27 * 3
        tensor_27 = None
        scale_25 = torch.sqrt(mul_29)
        mul_29 = None
        transpose_33 = pos_key_layer_8.transpose(-1, -2)
        pos_key_layer_8 = None
        c2p_att_16 = torch.bmm(query_layer_8, transpose_33)
        query_layer_8 = transpose_33 = None
        add_49 = relative_pos_17 + 256
        c2p_pos_8 = torch.clamp(add_49, 0, 511)
        add_49 = None
        squeeze_17 = c2p_pos_8.squeeze(0)
        c2p_pos_8 = None
        expand_16 = squeeze_17.expand([16, 11, 11])
        squeeze_17 = None
        c2p_att_17 = torch.gather(c2p_att_16, dim=-1, index=expand_16)
        c2p_att_16 = expand_16 = None
        to_36 = scale_25.to(dtype=torch.float32)
        scale_25 = None
        truediv_27 = c2p_att_17 / to_36
        c2p_att_17 = to_36 = None
        score_16 = 0 + truediv_27
        truediv_27 = None
        tensor_28 = torch.tensor(64, dtype=torch.float32)
        mul_30 = tensor_28 * 3
        tensor_28 = None
        scale_26 = torch.sqrt(mul_30)
        mul_30 = None
        neg_8 = -relative_pos_17
        relative_pos_17 = None
        add_51 = neg_8 + 256
        neg_8 = None
        p2c_pos_8 = torch.clamp(add_51, 0, 511)
        add_51 = None
        transpose_34 = pos_query_layer_8.transpose(-1, -2)
        pos_query_layer_8 = None
        p2c_att_16 = torch.bmm(key_layer_8, transpose_34)
        key_layer_8 = transpose_34 = None
        squeeze_18 = p2c_pos_8.squeeze(0)
        p2c_pos_8 = None
        expand_17 = squeeze_18.expand([16, 11, 11])
        squeeze_18 = None
        gather_17 = torch.gather(p2c_att_16, dim=-1, index=expand_17)
        p2c_att_16 = expand_17 = None
        p2c_att_17 = gather_17.transpose(-1, -2)
        gather_17 = None
        to_37 = scale_26.to(dtype=torch.float32)
        scale_26 = None
        truediv_28 = p2c_att_17 / to_37
        p2c_att_17 = to_37 = None
        score_16 += truediv_28
        score_17 = score_16
        score_16 = truediv_28 = None
        attention_scores_33 = attention_scores_32 + score_17
        attention_scores_32 = score_17 = None
        attention_scores_34 = attention_scores_33.view(-1, 16, 11, 11)
        attention_scores_33 = None
        attention_mask_9 = attention_mask.bool()
        invert_8 = ~attention_mask_9
        attention_mask_9 = None
        attention_scores_35 = attention_scores_34.masked_fill(
            invert_8, -3.4028234663852886e38
        )
        attention_scores_34 = invert_8 = None
        attention_probs_16 = torch.nn.functional.softmax(attention_scores_35, dim=-1)
        attention_scores_35 = None
        attention_probs_17 = torch.nn.functional.dropout(
            attention_probs_16, 0.1, False, False
        )
        attention_probs_16 = None
        view_123 = attention_probs_17.view(-1, 11, 11)
        attention_probs_17 = None
        context_layer_24 = torch.bmm(view_123, value_layer_8)
        view_123 = value_layer_8 = None
        view_124 = context_layer_24.view(-1, 16, 11, 64)
        context_layer_24 = None
        permute_53 = view_124.permute(0, 2, 1, 3)
        view_124 = None
        context_layer_25 = permute_53.contiguous()
        permute_53 = None
        context_layer_26 = context_layer_25.view((1, 11, -1))
        context_layer_25 = None
        hidden_states_64 = torch._C._nn.linear(
            context_layer_26,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_26 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_65 = torch.nn.functional.dropout(
            hidden_states_64, 0.1, False, False
        )
        hidden_states_64 = None
        add_53 = hidden_states_65 + hidden_states_63
        hidden_states_65 = hidden_states_63 = None
        hidden_states_66 = torch.nn.functional.layer_norm(
            add_53,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_53 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_67 = torch._C._nn.linear(
            hidden_states_66,
            l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_68 = torch._C._nn.gelu(hidden_states_67)
        hidden_states_67 = None
        hidden_states_69 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_68 = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, 0.1, False, False
        )
        hidden_states_69 = None
        add_54 = hidden_states_70 + hidden_states_66
        hidden_states_70 = hidden_states_66 = None
        hidden_states_71 = torch.nn.functional.layer_norm(
            add_54,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_54 = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_72 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_45 = linear_72.view((1, 11, 16, -1))
        linear_72 = None
        permute_54 = x_45.permute(0, 2, 1, 3)
        x_45 = None
        contiguous_54 = permute_54.contiguous()
        permute_54 = None
        query_layer_9 = contiguous_54.view(-1, 11, 64)
        contiguous_54 = None
        linear_73 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_46 = linear_73.view((1, 11, 16, -1))
        linear_73 = None
        permute_55 = x_46.permute(0, 2, 1, 3)
        x_46 = None
        contiguous_55 = permute_55.contiguous()
        permute_55 = None
        key_layer_9 = contiguous_55.view(-1, 11, 64)
        contiguous_55 = None
        linear_74 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_47 = linear_74.view((1, 11, 16, -1))
        linear_74 = None
        permute_56 = x_47.permute(0, 2, 1, 3)
        x_47 = None
        contiguous_56 = permute_56.contiguous()
        permute_56 = None
        value_layer_9 = contiguous_56.view(-1, 11, 64)
        contiguous_56 = None
        tensor_29 = torch.tensor(64, dtype=torch.float32)
        mul_31 = tensor_29 * 3
        tensor_29 = None
        scale_27 = torch.sqrt(mul_31)
        mul_31 = None
        transpose_36 = key_layer_9.transpose(-1, -2)
        to_38 = scale_27.to(dtype=torch.float32)
        scale_27 = None
        truediv_29 = transpose_36 / to_38
        transpose_36 = to_38 = None
        attention_scores_36 = torch.bmm(query_layer_9, truediv_29)
        truediv_29 = None
        rel_embeddings_19 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_18 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_19 = relative_pos_18.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_18 = None
        getitem_13 = rel_embeddings_19[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_19 = None
        rel_embeddings_20 = getitem_13.unsqueeze(0)
        getitem_13 = None
        linear_75 = torch._C._nn.linear(
            rel_embeddings_20,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_48 = linear_75.view((1, 512, 16, -1))
        linear_75 = None
        permute_57 = x_48.permute(0, 2, 1, 3)
        x_48 = None
        contiguous_57 = permute_57.contiguous()
        permute_57 = None
        view_133 = contiguous_57.view(-1, 512, 64)
        contiguous_57 = None
        pos_query_layer_9 = view_133.repeat(1, 1, 1)
        view_133 = None
        linear_76 = torch._C._nn.linear(
            rel_embeddings_20,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_20 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_49 = linear_76.view((1, 512, 16, -1))
        linear_76 = None
        permute_58 = x_49.permute(0, 2, 1, 3)
        x_49 = None
        contiguous_58 = permute_58.contiguous()
        permute_58 = None
        view_135 = contiguous_58.view(-1, 512, 64)
        contiguous_58 = None
        pos_key_layer_9 = view_135.repeat(1, 1, 1)
        view_135 = None
        tensor_30 = torch.tensor(64, dtype=torch.float32)
        mul_32 = tensor_30 * 3
        tensor_30 = None
        scale_28 = torch.sqrt(mul_32)
        mul_32 = None
        transpose_37 = pos_key_layer_9.transpose(-1, -2)
        pos_key_layer_9 = None
        c2p_att_18 = torch.bmm(query_layer_9, transpose_37)
        query_layer_9 = transpose_37 = None
        add_55 = relative_pos_19 + 256
        c2p_pos_9 = torch.clamp(add_55, 0, 511)
        add_55 = None
        squeeze_19 = c2p_pos_9.squeeze(0)
        c2p_pos_9 = None
        expand_18 = squeeze_19.expand([16, 11, 11])
        squeeze_19 = None
        c2p_att_19 = torch.gather(c2p_att_18, dim=-1, index=expand_18)
        c2p_att_18 = expand_18 = None
        to_40 = scale_28.to(dtype=torch.float32)
        scale_28 = None
        truediv_30 = c2p_att_19 / to_40
        c2p_att_19 = to_40 = None
        score_18 = 0 + truediv_30
        truediv_30 = None
        tensor_31 = torch.tensor(64, dtype=torch.float32)
        mul_33 = tensor_31 * 3
        tensor_31 = None
        scale_29 = torch.sqrt(mul_33)
        mul_33 = None
        neg_9 = -relative_pos_19
        relative_pos_19 = None
        add_57 = neg_9 + 256
        neg_9 = None
        p2c_pos_9 = torch.clamp(add_57, 0, 511)
        add_57 = None
        transpose_38 = pos_query_layer_9.transpose(-1, -2)
        pos_query_layer_9 = None
        p2c_att_18 = torch.bmm(key_layer_9, transpose_38)
        key_layer_9 = transpose_38 = None
        squeeze_20 = p2c_pos_9.squeeze(0)
        p2c_pos_9 = None
        expand_19 = squeeze_20.expand([16, 11, 11])
        squeeze_20 = None
        gather_19 = torch.gather(p2c_att_18, dim=-1, index=expand_19)
        p2c_att_18 = expand_19 = None
        p2c_att_19 = gather_19.transpose(-1, -2)
        gather_19 = None
        to_41 = scale_29.to(dtype=torch.float32)
        scale_29 = None
        truediv_31 = p2c_att_19 / to_41
        p2c_att_19 = to_41 = None
        score_18 += truediv_31
        score_19 = score_18
        score_18 = truediv_31 = None
        attention_scores_37 = attention_scores_36 + score_19
        attention_scores_36 = score_19 = None
        attention_scores_38 = attention_scores_37.view(-1, 16, 11, 11)
        attention_scores_37 = None
        attention_mask_10 = attention_mask.bool()
        invert_9 = ~attention_mask_10
        attention_mask_10 = None
        attention_scores_39 = attention_scores_38.masked_fill(
            invert_9, -3.4028234663852886e38
        )
        attention_scores_38 = invert_9 = None
        attention_probs_18 = torch.nn.functional.softmax(attention_scores_39, dim=-1)
        attention_scores_39 = None
        attention_probs_19 = torch.nn.functional.dropout(
            attention_probs_18, 0.1, False, False
        )
        attention_probs_18 = None
        view_137 = attention_probs_19.view(-1, 11, 11)
        attention_probs_19 = None
        context_layer_27 = torch.bmm(view_137, value_layer_9)
        view_137 = value_layer_9 = None
        view_138 = context_layer_27.view(-1, 16, 11, 64)
        context_layer_27 = None
        permute_59 = view_138.permute(0, 2, 1, 3)
        view_138 = None
        context_layer_28 = permute_59.contiguous()
        permute_59 = None
        context_layer_29 = context_layer_28.view((1, 11, -1))
        context_layer_28 = None
        hidden_states_72 = torch._C._nn.linear(
            context_layer_29,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_29 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_73 = torch.nn.functional.dropout(
            hidden_states_72, 0.1, False, False
        )
        hidden_states_72 = None
        add_59 = hidden_states_73 + hidden_states_71
        hidden_states_73 = hidden_states_71 = None
        hidden_states_74 = torch.nn.functional.layer_norm(
            add_59,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_59 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_75 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_76 = torch._C._nn.gelu(hidden_states_75)
        hidden_states_75 = None
        hidden_states_77 = torch._C._nn.linear(
            hidden_states_76,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_76 = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_78 = torch.nn.functional.dropout(
            hidden_states_77, 0.1, False, False
        )
        hidden_states_77 = None
        add_60 = hidden_states_78 + hidden_states_74
        hidden_states_78 = hidden_states_74 = None
        hidden_states_79 = torch.nn.functional.layer_norm(
            add_60,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_60 = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_80 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_50 = linear_80.view((1, 11, 16, -1))
        linear_80 = None
        permute_60 = x_50.permute(0, 2, 1, 3)
        x_50 = None
        contiguous_60 = permute_60.contiguous()
        permute_60 = None
        query_layer_10 = contiguous_60.view(-1, 11, 64)
        contiguous_60 = None
        linear_81 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_51 = linear_81.view((1, 11, 16, -1))
        linear_81 = None
        permute_61 = x_51.permute(0, 2, 1, 3)
        x_51 = None
        contiguous_61 = permute_61.contiguous()
        permute_61 = None
        key_layer_10 = contiguous_61.view(-1, 11, 64)
        contiguous_61 = None
        linear_82 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_52 = linear_82.view((1, 11, 16, -1))
        linear_82 = None
        permute_62 = x_52.permute(0, 2, 1, 3)
        x_52 = None
        contiguous_62 = permute_62.contiguous()
        permute_62 = None
        value_layer_10 = contiguous_62.view(-1, 11, 64)
        contiguous_62 = None
        tensor_32 = torch.tensor(64, dtype=torch.float32)
        mul_34 = tensor_32 * 3
        tensor_32 = None
        scale_30 = torch.sqrt(mul_34)
        mul_34 = None
        transpose_40 = key_layer_10.transpose(-1, -2)
        to_42 = scale_30.to(dtype=torch.float32)
        scale_30 = None
        truediv_32 = transpose_40 / to_42
        transpose_40 = to_42 = None
        attention_scores_40 = torch.bmm(query_layer_10, truediv_32)
        truediv_32 = None
        rel_embeddings_21 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_20 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_21 = relative_pos_20.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_20 = None
        getitem_14 = rel_embeddings_21[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_21 = None
        rel_embeddings_22 = getitem_14.unsqueeze(0)
        getitem_14 = None
        linear_83 = torch._C._nn.linear(
            rel_embeddings_22,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_53 = linear_83.view((1, 512, 16, -1))
        linear_83 = None
        permute_63 = x_53.permute(0, 2, 1, 3)
        x_53 = None
        contiguous_63 = permute_63.contiguous()
        permute_63 = None
        view_147 = contiguous_63.view(-1, 512, 64)
        contiguous_63 = None
        pos_query_layer_10 = view_147.repeat(1, 1, 1)
        view_147 = None
        linear_84 = torch._C._nn.linear(
            rel_embeddings_22,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_22 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_54 = linear_84.view((1, 512, 16, -1))
        linear_84 = None
        permute_64 = x_54.permute(0, 2, 1, 3)
        x_54 = None
        contiguous_64 = permute_64.contiguous()
        permute_64 = None
        view_149 = contiguous_64.view(-1, 512, 64)
        contiguous_64 = None
        pos_key_layer_10 = view_149.repeat(1, 1, 1)
        view_149 = None
        tensor_33 = torch.tensor(64, dtype=torch.float32)
        mul_35 = tensor_33 * 3
        tensor_33 = None
        scale_31 = torch.sqrt(mul_35)
        mul_35 = None
        transpose_41 = pos_key_layer_10.transpose(-1, -2)
        pos_key_layer_10 = None
        c2p_att_20 = torch.bmm(query_layer_10, transpose_41)
        query_layer_10 = transpose_41 = None
        add_61 = relative_pos_21 + 256
        c2p_pos_10 = torch.clamp(add_61, 0, 511)
        add_61 = None
        squeeze_21 = c2p_pos_10.squeeze(0)
        c2p_pos_10 = None
        expand_20 = squeeze_21.expand([16, 11, 11])
        squeeze_21 = None
        c2p_att_21 = torch.gather(c2p_att_20, dim=-1, index=expand_20)
        c2p_att_20 = expand_20 = None
        to_44 = scale_31.to(dtype=torch.float32)
        scale_31 = None
        truediv_33 = c2p_att_21 / to_44
        c2p_att_21 = to_44 = None
        score_20 = 0 + truediv_33
        truediv_33 = None
        tensor_34 = torch.tensor(64, dtype=torch.float32)
        mul_36 = tensor_34 * 3
        tensor_34 = None
        scale_32 = torch.sqrt(mul_36)
        mul_36 = None
        neg_10 = -relative_pos_21
        relative_pos_21 = None
        add_63 = neg_10 + 256
        neg_10 = None
        p2c_pos_10 = torch.clamp(add_63, 0, 511)
        add_63 = None
        transpose_42 = pos_query_layer_10.transpose(-1, -2)
        pos_query_layer_10 = None
        p2c_att_20 = torch.bmm(key_layer_10, transpose_42)
        key_layer_10 = transpose_42 = None
        squeeze_22 = p2c_pos_10.squeeze(0)
        p2c_pos_10 = None
        expand_21 = squeeze_22.expand([16, 11, 11])
        squeeze_22 = None
        gather_21 = torch.gather(p2c_att_20, dim=-1, index=expand_21)
        p2c_att_20 = expand_21 = None
        p2c_att_21 = gather_21.transpose(-1, -2)
        gather_21 = None
        to_45 = scale_32.to(dtype=torch.float32)
        scale_32 = None
        truediv_34 = p2c_att_21 / to_45
        p2c_att_21 = to_45 = None
        score_20 += truediv_34
        score_21 = score_20
        score_20 = truediv_34 = None
        attention_scores_41 = attention_scores_40 + score_21
        attention_scores_40 = score_21 = None
        attention_scores_42 = attention_scores_41.view(-1, 16, 11, 11)
        attention_scores_41 = None
        attention_mask_11 = attention_mask.bool()
        invert_10 = ~attention_mask_11
        attention_mask_11 = None
        attention_scores_43 = attention_scores_42.masked_fill(
            invert_10, -3.4028234663852886e38
        )
        attention_scores_42 = invert_10 = None
        attention_probs_20 = torch.nn.functional.softmax(attention_scores_43, dim=-1)
        attention_scores_43 = None
        attention_probs_21 = torch.nn.functional.dropout(
            attention_probs_20, 0.1, False, False
        )
        attention_probs_20 = None
        view_151 = attention_probs_21.view(-1, 11, 11)
        attention_probs_21 = None
        context_layer_30 = torch.bmm(view_151, value_layer_10)
        view_151 = value_layer_10 = None
        view_152 = context_layer_30.view(-1, 16, 11, 64)
        context_layer_30 = None
        permute_65 = view_152.permute(0, 2, 1, 3)
        view_152 = None
        context_layer_31 = permute_65.contiguous()
        permute_65 = None
        context_layer_32 = context_layer_31.view((1, 11, -1))
        context_layer_31 = None
        hidden_states_80 = torch._C._nn.linear(
            context_layer_32,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_32 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_81 = torch.nn.functional.dropout(
            hidden_states_80, 0.1, False, False
        )
        hidden_states_80 = None
        add_65 = hidden_states_81 + hidden_states_79
        hidden_states_81 = hidden_states_79 = None
        hidden_states_82 = torch.nn.functional.layer_norm(
            add_65,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_65 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_83 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_84 = torch._C._nn.gelu(hidden_states_83)
        hidden_states_83 = None
        hidden_states_85 = torch._C._nn.linear(
            hidden_states_84,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_84 = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_86 = torch.nn.functional.dropout(
            hidden_states_85, 0.1, False, False
        )
        hidden_states_85 = None
        add_66 = hidden_states_86 + hidden_states_82
        hidden_states_86 = hidden_states_82 = None
        hidden_states_87 = torch.nn.functional.layer_norm(
            add_66,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_66 = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_88 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_55 = linear_88.view((1, 11, 16, -1))
        linear_88 = None
        permute_66 = x_55.permute(0, 2, 1, 3)
        x_55 = None
        contiguous_66 = permute_66.contiguous()
        permute_66 = None
        query_layer_11 = contiguous_66.view(-1, 11, 64)
        contiguous_66 = None
        linear_89 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_56 = linear_89.view((1, 11, 16, -1))
        linear_89 = None
        permute_67 = x_56.permute(0, 2, 1, 3)
        x_56 = None
        contiguous_67 = permute_67.contiguous()
        permute_67 = None
        key_layer_11 = contiguous_67.view(-1, 11, 64)
        contiguous_67 = None
        linear_90 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_57 = linear_90.view((1, 11, 16, -1))
        linear_90 = None
        permute_68 = x_57.permute(0, 2, 1, 3)
        x_57 = None
        contiguous_68 = permute_68.contiguous()
        permute_68 = None
        value_layer_11 = contiguous_68.view(-1, 11, 64)
        contiguous_68 = None
        tensor_35 = torch.tensor(64, dtype=torch.float32)
        mul_37 = tensor_35 * 3
        tensor_35 = None
        scale_33 = torch.sqrt(mul_37)
        mul_37 = None
        transpose_44 = key_layer_11.transpose(-1, -2)
        to_46 = scale_33.to(dtype=torch.float32)
        scale_33 = None
        truediv_35 = transpose_44 / to_46
        transpose_44 = to_46 = None
        attention_scores_44 = torch.bmm(query_layer_11, truediv_35)
        truediv_35 = None
        rel_embeddings_23 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_22 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_23 = relative_pos_22.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_22 = None
        getitem_15 = rel_embeddings_23[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_23 = None
        rel_embeddings_24 = getitem_15.unsqueeze(0)
        getitem_15 = None
        linear_91 = torch._C._nn.linear(
            rel_embeddings_24,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_58 = linear_91.view((1, 512, 16, -1))
        linear_91 = None
        permute_69 = x_58.permute(0, 2, 1, 3)
        x_58 = None
        contiguous_69 = permute_69.contiguous()
        permute_69 = None
        view_161 = contiguous_69.view(-1, 512, 64)
        contiguous_69 = None
        pos_query_layer_11 = view_161.repeat(1, 1, 1)
        view_161 = None
        linear_92 = torch._C._nn.linear(
            rel_embeddings_24,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_24 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_59 = linear_92.view((1, 512, 16, -1))
        linear_92 = None
        permute_70 = x_59.permute(0, 2, 1, 3)
        x_59 = None
        contiguous_70 = permute_70.contiguous()
        permute_70 = None
        view_163 = contiguous_70.view(-1, 512, 64)
        contiguous_70 = None
        pos_key_layer_11 = view_163.repeat(1, 1, 1)
        view_163 = None
        tensor_36 = torch.tensor(64, dtype=torch.float32)
        mul_38 = tensor_36 * 3
        tensor_36 = None
        scale_34 = torch.sqrt(mul_38)
        mul_38 = None
        transpose_45 = pos_key_layer_11.transpose(-1, -2)
        pos_key_layer_11 = None
        c2p_att_22 = torch.bmm(query_layer_11, transpose_45)
        query_layer_11 = transpose_45 = None
        add_67 = relative_pos_23 + 256
        c2p_pos_11 = torch.clamp(add_67, 0, 511)
        add_67 = None
        squeeze_23 = c2p_pos_11.squeeze(0)
        c2p_pos_11 = None
        expand_22 = squeeze_23.expand([16, 11, 11])
        squeeze_23 = None
        c2p_att_23 = torch.gather(c2p_att_22, dim=-1, index=expand_22)
        c2p_att_22 = expand_22 = None
        to_48 = scale_34.to(dtype=torch.float32)
        scale_34 = None
        truediv_36 = c2p_att_23 / to_48
        c2p_att_23 = to_48 = None
        score_22 = 0 + truediv_36
        truediv_36 = None
        tensor_37 = torch.tensor(64, dtype=torch.float32)
        mul_39 = tensor_37 * 3
        tensor_37 = None
        scale_35 = torch.sqrt(mul_39)
        mul_39 = None
        neg_11 = -relative_pos_23
        relative_pos_23 = None
        add_69 = neg_11 + 256
        neg_11 = None
        p2c_pos_11 = torch.clamp(add_69, 0, 511)
        add_69 = None
        transpose_46 = pos_query_layer_11.transpose(-1, -2)
        pos_query_layer_11 = None
        p2c_att_22 = torch.bmm(key_layer_11, transpose_46)
        key_layer_11 = transpose_46 = None
        squeeze_24 = p2c_pos_11.squeeze(0)
        p2c_pos_11 = None
        expand_23 = squeeze_24.expand([16, 11, 11])
        squeeze_24 = None
        gather_23 = torch.gather(p2c_att_22, dim=-1, index=expand_23)
        p2c_att_22 = expand_23 = None
        p2c_att_23 = gather_23.transpose(-1, -2)
        gather_23 = None
        to_49 = scale_35.to(dtype=torch.float32)
        scale_35 = None
        truediv_37 = p2c_att_23 / to_49
        p2c_att_23 = to_49 = None
        score_22 += truediv_37
        score_23 = score_22
        score_22 = truediv_37 = None
        attention_scores_45 = attention_scores_44 + score_23
        attention_scores_44 = score_23 = None
        attention_scores_46 = attention_scores_45.view(-1, 16, 11, 11)
        attention_scores_45 = None
        attention_mask_12 = attention_mask.bool()
        invert_11 = ~attention_mask_12
        attention_mask_12 = None
        attention_scores_47 = attention_scores_46.masked_fill(
            invert_11, -3.4028234663852886e38
        )
        attention_scores_46 = invert_11 = None
        attention_probs_22 = torch.nn.functional.softmax(attention_scores_47, dim=-1)
        attention_scores_47 = None
        attention_probs_23 = torch.nn.functional.dropout(
            attention_probs_22, 0.1, False, False
        )
        attention_probs_22 = None
        view_165 = attention_probs_23.view(-1, 11, 11)
        attention_probs_23 = None
        context_layer_33 = torch.bmm(view_165, value_layer_11)
        view_165 = value_layer_11 = None
        view_166 = context_layer_33.view(-1, 16, 11, 64)
        context_layer_33 = None
        permute_71 = view_166.permute(0, 2, 1, 3)
        view_166 = None
        context_layer_34 = permute_71.contiguous()
        permute_71 = None
        context_layer_35 = context_layer_34.view((1, 11, -1))
        context_layer_34 = None
        hidden_states_88 = torch._C._nn.linear(
            context_layer_35,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_35 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_89 = torch.nn.functional.dropout(
            hidden_states_88, 0.1, False, False
        )
        hidden_states_88 = None
        add_71 = hidden_states_89 + hidden_states_87
        hidden_states_89 = hidden_states_87 = None
        hidden_states_90 = torch.nn.functional.layer_norm(
            add_71,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_71 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_91 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_92 = torch._C._nn.gelu(hidden_states_91)
        hidden_states_91 = None
        hidden_states_93 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_94 = torch.nn.functional.dropout(
            hidden_states_93, 0.1, False, False
        )
        hidden_states_93 = None
        add_72 = hidden_states_94 + hidden_states_90
        hidden_states_94 = hidden_states_90 = None
        hidden_states_95 = torch.nn.functional.layer_norm(
            add_72,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_72 = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_96 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_60 = linear_96.view((1, 11, 16, -1))
        linear_96 = None
        permute_72 = x_60.permute(0, 2, 1, 3)
        x_60 = None
        contiguous_72 = permute_72.contiguous()
        permute_72 = None
        query_layer_12 = contiguous_72.view(-1, 11, 64)
        contiguous_72 = None
        linear_97 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_61 = linear_97.view((1, 11, 16, -1))
        linear_97 = None
        permute_73 = x_61.permute(0, 2, 1, 3)
        x_61 = None
        contiguous_73 = permute_73.contiguous()
        permute_73 = None
        key_layer_12 = contiguous_73.view(-1, 11, 64)
        contiguous_73 = None
        linear_98 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_62 = linear_98.view((1, 11, 16, -1))
        linear_98 = None
        permute_74 = x_62.permute(0, 2, 1, 3)
        x_62 = None
        contiguous_74 = permute_74.contiguous()
        permute_74 = None
        value_layer_12 = contiguous_74.view(-1, 11, 64)
        contiguous_74 = None
        tensor_38 = torch.tensor(64, dtype=torch.float32)
        mul_40 = tensor_38 * 3
        tensor_38 = None
        scale_36 = torch.sqrt(mul_40)
        mul_40 = None
        transpose_48 = key_layer_12.transpose(-1, -2)
        to_50 = scale_36.to(dtype=torch.float32)
        scale_36 = None
        truediv_38 = transpose_48 / to_50
        transpose_48 = to_50 = None
        attention_scores_48 = torch.bmm(query_layer_12, truediv_38)
        truediv_38 = None
        rel_embeddings_25 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_24 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_25 = relative_pos_24.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_24 = None
        getitem_16 = rel_embeddings_25[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_25 = None
        rel_embeddings_26 = getitem_16.unsqueeze(0)
        getitem_16 = None
        linear_99 = torch._C._nn.linear(
            rel_embeddings_26,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_63 = linear_99.view((1, 512, 16, -1))
        linear_99 = None
        permute_75 = x_63.permute(0, 2, 1, 3)
        x_63 = None
        contiguous_75 = permute_75.contiguous()
        permute_75 = None
        view_175 = contiguous_75.view(-1, 512, 64)
        contiguous_75 = None
        pos_query_layer_12 = view_175.repeat(1, 1, 1)
        view_175 = None
        linear_100 = torch._C._nn.linear(
            rel_embeddings_26,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_26 = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_64 = linear_100.view((1, 512, 16, -1))
        linear_100 = None
        permute_76 = x_64.permute(0, 2, 1, 3)
        x_64 = None
        contiguous_76 = permute_76.contiguous()
        permute_76 = None
        view_177 = contiguous_76.view(-1, 512, 64)
        contiguous_76 = None
        pos_key_layer_12 = view_177.repeat(1, 1, 1)
        view_177 = None
        tensor_39 = torch.tensor(64, dtype=torch.float32)
        mul_41 = tensor_39 * 3
        tensor_39 = None
        scale_37 = torch.sqrt(mul_41)
        mul_41 = None
        transpose_49 = pos_key_layer_12.transpose(-1, -2)
        pos_key_layer_12 = None
        c2p_att_24 = torch.bmm(query_layer_12, transpose_49)
        query_layer_12 = transpose_49 = None
        add_73 = relative_pos_25 + 256
        c2p_pos_12 = torch.clamp(add_73, 0, 511)
        add_73 = None
        squeeze_25 = c2p_pos_12.squeeze(0)
        c2p_pos_12 = None
        expand_24 = squeeze_25.expand([16, 11, 11])
        squeeze_25 = None
        c2p_att_25 = torch.gather(c2p_att_24, dim=-1, index=expand_24)
        c2p_att_24 = expand_24 = None
        to_52 = scale_37.to(dtype=torch.float32)
        scale_37 = None
        truediv_39 = c2p_att_25 / to_52
        c2p_att_25 = to_52 = None
        score_24 = 0 + truediv_39
        truediv_39 = None
        tensor_40 = torch.tensor(64, dtype=torch.float32)
        mul_42 = tensor_40 * 3
        tensor_40 = None
        scale_38 = torch.sqrt(mul_42)
        mul_42 = None
        neg_12 = -relative_pos_25
        relative_pos_25 = None
        add_75 = neg_12 + 256
        neg_12 = None
        p2c_pos_12 = torch.clamp(add_75, 0, 511)
        add_75 = None
        transpose_50 = pos_query_layer_12.transpose(-1, -2)
        pos_query_layer_12 = None
        p2c_att_24 = torch.bmm(key_layer_12, transpose_50)
        key_layer_12 = transpose_50 = None
        squeeze_26 = p2c_pos_12.squeeze(0)
        p2c_pos_12 = None
        expand_25 = squeeze_26.expand([16, 11, 11])
        squeeze_26 = None
        gather_25 = torch.gather(p2c_att_24, dim=-1, index=expand_25)
        p2c_att_24 = expand_25 = None
        p2c_att_25 = gather_25.transpose(-1, -2)
        gather_25 = None
        to_53 = scale_38.to(dtype=torch.float32)
        scale_38 = None
        truediv_40 = p2c_att_25 / to_53
        p2c_att_25 = to_53 = None
        score_24 += truediv_40
        score_25 = score_24
        score_24 = truediv_40 = None
        attention_scores_49 = attention_scores_48 + score_25
        attention_scores_48 = score_25 = None
        attention_scores_50 = attention_scores_49.view(-1, 16, 11, 11)
        attention_scores_49 = None
        attention_mask_13 = attention_mask.bool()
        invert_12 = ~attention_mask_13
        attention_mask_13 = None
        attention_scores_51 = attention_scores_50.masked_fill(
            invert_12, -3.4028234663852886e38
        )
        attention_scores_50 = invert_12 = None
        attention_probs_24 = torch.nn.functional.softmax(attention_scores_51, dim=-1)
        attention_scores_51 = None
        attention_probs_25 = torch.nn.functional.dropout(
            attention_probs_24, 0.1, False, False
        )
        attention_probs_24 = None
        view_179 = attention_probs_25.view(-1, 11, 11)
        attention_probs_25 = None
        context_layer_36 = torch.bmm(view_179, value_layer_12)
        view_179 = value_layer_12 = None
        view_180 = context_layer_36.view(-1, 16, 11, 64)
        context_layer_36 = None
        permute_77 = view_180.permute(0, 2, 1, 3)
        view_180 = None
        context_layer_37 = permute_77.contiguous()
        permute_77 = None
        context_layer_38 = context_layer_37.view((1, 11, -1))
        context_layer_37 = None
        hidden_states_96 = torch._C._nn.linear(
            context_layer_38,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_38 = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_97 = torch.nn.functional.dropout(
            hidden_states_96, 0.1, False, False
        )
        hidden_states_96 = None
        add_77 = hidden_states_97 + hidden_states_95
        hidden_states_97 = hidden_states_95 = None
        hidden_states_98 = torch.nn.functional.layer_norm(
            add_77,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_77 = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_99 = torch._C._nn.linear(
            hidden_states_98,
            l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_100 = torch._C._nn.gelu(hidden_states_99)
        hidden_states_99 = None
        hidden_states_101 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_100 = l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_102 = torch.nn.functional.dropout(
            hidden_states_101, 0.1, False, False
        )
        hidden_states_101 = None
        add_78 = hidden_states_102 + hidden_states_98
        hidden_states_102 = hidden_states_98 = None
        hidden_states_103 = torch.nn.functional.layer_norm(
            add_78,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_78 = l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_104 = torch._C._nn.linear(
            hidden_states_103,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_65 = linear_104.view((1, 11, 16, -1))
        linear_104 = None
        permute_78 = x_65.permute(0, 2, 1, 3)
        x_65 = None
        contiguous_78 = permute_78.contiguous()
        permute_78 = None
        query_layer_13 = contiguous_78.view(-1, 11, 64)
        contiguous_78 = None
        linear_105 = torch._C._nn.linear(
            hidden_states_103,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_66 = linear_105.view((1, 11, 16, -1))
        linear_105 = None
        permute_79 = x_66.permute(0, 2, 1, 3)
        x_66 = None
        contiguous_79 = permute_79.contiguous()
        permute_79 = None
        key_layer_13 = contiguous_79.view(-1, 11, 64)
        contiguous_79 = None
        linear_106 = torch._C._nn.linear(
            hidden_states_103,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_67 = linear_106.view((1, 11, 16, -1))
        linear_106 = None
        permute_80 = x_67.permute(0, 2, 1, 3)
        x_67 = None
        contiguous_80 = permute_80.contiguous()
        permute_80 = None
        value_layer_13 = contiguous_80.view(-1, 11, 64)
        contiguous_80 = None
        tensor_41 = torch.tensor(64, dtype=torch.float32)
        mul_43 = tensor_41 * 3
        tensor_41 = None
        scale_39 = torch.sqrt(mul_43)
        mul_43 = None
        transpose_52 = key_layer_13.transpose(-1, -2)
        to_54 = scale_39.to(dtype=torch.float32)
        scale_39 = None
        truediv_41 = transpose_52 / to_54
        transpose_52 = to_54 = None
        attention_scores_52 = torch.bmm(query_layer_13, truediv_41)
        truediv_41 = None
        rel_embeddings_27 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_26 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_27 = relative_pos_26.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_26 = None
        getitem_17 = rel_embeddings_27[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_27 = None
        rel_embeddings_28 = getitem_17.unsqueeze(0)
        getitem_17 = None
        linear_107 = torch._C._nn.linear(
            rel_embeddings_28,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_68 = linear_107.view((1, 512, 16, -1))
        linear_107 = None
        permute_81 = x_68.permute(0, 2, 1, 3)
        x_68 = None
        contiguous_81 = permute_81.contiguous()
        permute_81 = None
        view_189 = contiguous_81.view(-1, 512, 64)
        contiguous_81 = None
        pos_query_layer_13 = view_189.repeat(1, 1, 1)
        view_189 = None
        linear_108 = torch._C._nn.linear(
            rel_embeddings_28,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_28 = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_69 = linear_108.view((1, 512, 16, -1))
        linear_108 = None
        permute_82 = x_69.permute(0, 2, 1, 3)
        x_69 = None
        contiguous_82 = permute_82.contiguous()
        permute_82 = None
        view_191 = contiguous_82.view(-1, 512, 64)
        contiguous_82 = None
        pos_key_layer_13 = view_191.repeat(1, 1, 1)
        view_191 = None
        tensor_42 = torch.tensor(64, dtype=torch.float32)
        mul_44 = tensor_42 * 3
        tensor_42 = None
        scale_40 = torch.sqrt(mul_44)
        mul_44 = None
        transpose_53 = pos_key_layer_13.transpose(-1, -2)
        pos_key_layer_13 = None
        c2p_att_26 = torch.bmm(query_layer_13, transpose_53)
        query_layer_13 = transpose_53 = None
        add_79 = relative_pos_27 + 256
        c2p_pos_13 = torch.clamp(add_79, 0, 511)
        add_79 = None
        squeeze_27 = c2p_pos_13.squeeze(0)
        c2p_pos_13 = None
        expand_26 = squeeze_27.expand([16, 11, 11])
        squeeze_27 = None
        c2p_att_27 = torch.gather(c2p_att_26, dim=-1, index=expand_26)
        c2p_att_26 = expand_26 = None
        to_56 = scale_40.to(dtype=torch.float32)
        scale_40 = None
        truediv_42 = c2p_att_27 / to_56
        c2p_att_27 = to_56 = None
        score_26 = 0 + truediv_42
        truediv_42 = None
        tensor_43 = torch.tensor(64, dtype=torch.float32)
        mul_45 = tensor_43 * 3
        tensor_43 = None
        scale_41 = torch.sqrt(mul_45)
        mul_45 = None
        neg_13 = -relative_pos_27
        relative_pos_27 = None
        add_81 = neg_13 + 256
        neg_13 = None
        p2c_pos_13 = torch.clamp(add_81, 0, 511)
        add_81 = None
        transpose_54 = pos_query_layer_13.transpose(-1, -2)
        pos_query_layer_13 = None
        p2c_att_26 = torch.bmm(key_layer_13, transpose_54)
        key_layer_13 = transpose_54 = None
        squeeze_28 = p2c_pos_13.squeeze(0)
        p2c_pos_13 = None
        expand_27 = squeeze_28.expand([16, 11, 11])
        squeeze_28 = None
        gather_27 = torch.gather(p2c_att_26, dim=-1, index=expand_27)
        p2c_att_26 = expand_27 = None
        p2c_att_27 = gather_27.transpose(-1, -2)
        gather_27 = None
        to_57 = scale_41.to(dtype=torch.float32)
        scale_41 = None
        truediv_43 = p2c_att_27 / to_57
        p2c_att_27 = to_57 = None
        score_26 += truediv_43
        score_27 = score_26
        score_26 = truediv_43 = None
        attention_scores_53 = attention_scores_52 + score_27
        attention_scores_52 = score_27 = None
        attention_scores_54 = attention_scores_53.view(-1, 16, 11, 11)
        attention_scores_53 = None
        attention_mask_14 = attention_mask.bool()
        invert_13 = ~attention_mask_14
        attention_mask_14 = None
        attention_scores_55 = attention_scores_54.masked_fill(
            invert_13, -3.4028234663852886e38
        )
        attention_scores_54 = invert_13 = None
        attention_probs_26 = torch.nn.functional.softmax(attention_scores_55, dim=-1)
        attention_scores_55 = None
        attention_probs_27 = torch.nn.functional.dropout(
            attention_probs_26, 0.1, False, False
        )
        attention_probs_26 = None
        view_193 = attention_probs_27.view(-1, 11, 11)
        attention_probs_27 = None
        context_layer_39 = torch.bmm(view_193, value_layer_13)
        view_193 = value_layer_13 = None
        view_194 = context_layer_39.view(-1, 16, 11, 64)
        context_layer_39 = None
        permute_83 = view_194.permute(0, 2, 1, 3)
        view_194 = None
        context_layer_40 = permute_83.contiguous()
        permute_83 = None
        context_layer_41 = context_layer_40.view((1, 11, -1))
        context_layer_40 = None
        hidden_states_104 = torch._C._nn.linear(
            context_layer_41,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_41 = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_105 = torch.nn.functional.dropout(
            hidden_states_104, 0.1, False, False
        )
        hidden_states_104 = None
        add_83 = hidden_states_105 + hidden_states_103
        hidden_states_105 = hidden_states_103 = None
        hidden_states_106 = torch.nn.functional.layer_norm(
            add_83,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_83 = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_107 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_108 = torch._C._nn.gelu(hidden_states_107)
        hidden_states_107 = None
        hidden_states_109 = torch._C._nn.linear(
            hidden_states_108,
            l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_108 = l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_110 = torch.nn.functional.dropout(
            hidden_states_109, 0.1, False, False
        )
        hidden_states_109 = None
        add_84 = hidden_states_110 + hidden_states_106
        hidden_states_110 = hidden_states_106 = None
        hidden_states_111 = torch.nn.functional.layer_norm(
            add_84,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_84 = l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_112 = torch._C._nn.linear(
            hidden_states_111,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_70 = linear_112.view((1, 11, 16, -1))
        linear_112 = None
        permute_84 = x_70.permute(0, 2, 1, 3)
        x_70 = None
        contiguous_84 = permute_84.contiguous()
        permute_84 = None
        query_layer_14 = contiguous_84.view(-1, 11, 64)
        contiguous_84 = None
        linear_113 = torch._C._nn.linear(
            hidden_states_111,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_71 = linear_113.view((1, 11, 16, -1))
        linear_113 = None
        permute_85 = x_71.permute(0, 2, 1, 3)
        x_71 = None
        contiguous_85 = permute_85.contiguous()
        permute_85 = None
        key_layer_14 = contiguous_85.view(-1, 11, 64)
        contiguous_85 = None
        linear_114 = torch._C._nn.linear(
            hidden_states_111,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_72 = linear_114.view((1, 11, 16, -1))
        linear_114 = None
        permute_86 = x_72.permute(0, 2, 1, 3)
        x_72 = None
        contiguous_86 = permute_86.contiguous()
        permute_86 = None
        value_layer_14 = contiguous_86.view(-1, 11, 64)
        contiguous_86 = None
        tensor_44 = torch.tensor(64, dtype=torch.float32)
        mul_46 = tensor_44 * 3
        tensor_44 = None
        scale_42 = torch.sqrt(mul_46)
        mul_46 = None
        transpose_56 = key_layer_14.transpose(-1, -2)
        to_58 = scale_42.to(dtype=torch.float32)
        scale_42 = None
        truediv_44 = transpose_56 / to_58
        transpose_56 = to_58 = None
        attention_scores_56 = torch.bmm(query_layer_14, truediv_44)
        truediv_44 = None
        rel_embeddings_29 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_28 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_29 = relative_pos_28.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_28 = None
        getitem_18 = rel_embeddings_29[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_29 = None
        rel_embeddings_30 = getitem_18.unsqueeze(0)
        getitem_18 = None
        linear_115 = torch._C._nn.linear(
            rel_embeddings_30,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_73 = linear_115.view((1, 512, 16, -1))
        linear_115 = None
        permute_87 = x_73.permute(0, 2, 1, 3)
        x_73 = None
        contiguous_87 = permute_87.contiguous()
        permute_87 = None
        view_203 = contiguous_87.view(-1, 512, 64)
        contiguous_87 = None
        pos_query_layer_14 = view_203.repeat(1, 1, 1)
        view_203 = None
        linear_116 = torch._C._nn.linear(
            rel_embeddings_30,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_30 = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_74 = linear_116.view((1, 512, 16, -1))
        linear_116 = None
        permute_88 = x_74.permute(0, 2, 1, 3)
        x_74 = None
        contiguous_88 = permute_88.contiguous()
        permute_88 = None
        view_205 = contiguous_88.view(-1, 512, 64)
        contiguous_88 = None
        pos_key_layer_14 = view_205.repeat(1, 1, 1)
        view_205 = None
        tensor_45 = torch.tensor(64, dtype=torch.float32)
        mul_47 = tensor_45 * 3
        tensor_45 = None
        scale_43 = torch.sqrt(mul_47)
        mul_47 = None
        transpose_57 = pos_key_layer_14.transpose(-1, -2)
        pos_key_layer_14 = None
        c2p_att_28 = torch.bmm(query_layer_14, transpose_57)
        query_layer_14 = transpose_57 = None
        add_85 = relative_pos_29 + 256
        c2p_pos_14 = torch.clamp(add_85, 0, 511)
        add_85 = None
        squeeze_29 = c2p_pos_14.squeeze(0)
        c2p_pos_14 = None
        expand_28 = squeeze_29.expand([16, 11, 11])
        squeeze_29 = None
        c2p_att_29 = torch.gather(c2p_att_28, dim=-1, index=expand_28)
        c2p_att_28 = expand_28 = None
        to_60 = scale_43.to(dtype=torch.float32)
        scale_43 = None
        truediv_45 = c2p_att_29 / to_60
        c2p_att_29 = to_60 = None
        score_28 = 0 + truediv_45
        truediv_45 = None
        tensor_46 = torch.tensor(64, dtype=torch.float32)
        mul_48 = tensor_46 * 3
        tensor_46 = None
        scale_44 = torch.sqrt(mul_48)
        mul_48 = None
        neg_14 = -relative_pos_29
        relative_pos_29 = None
        add_87 = neg_14 + 256
        neg_14 = None
        p2c_pos_14 = torch.clamp(add_87, 0, 511)
        add_87 = None
        transpose_58 = pos_query_layer_14.transpose(-1, -2)
        pos_query_layer_14 = None
        p2c_att_28 = torch.bmm(key_layer_14, transpose_58)
        key_layer_14 = transpose_58 = None
        squeeze_30 = p2c_pos_14.squeeze(0)
        p2c_pos_14 = None
        expand_29 = squeeze_30.expand([16, 11, 11])
        squeeze_30 = None
        gather_29 = torch.gather(p2c_att_28, dim=-1, index=expand_29)
        p2c_att_28 = expand_29 = None
        p2c_att_29 = gather_29.transpose(-1, -2)
        gather_29 = None
        to_61 = scale_44.to(dtype=torch.float32)
        scale_44 = None
        truediv_46 = p2c_att_29 / to_61
        p2c_att_29 = to_61 = None
        score_28 += truediv_46
        score_29 = score_28
        score_28 = truediv_46 = None
        attention_scores_57 = attention_scores_56 + score_29
        attention_scores_56 = score_29 = None
        attention_scores_58 = attention_scores_57.view(-1, 16, 11, 11)
        attention_scores_57 = None
        attention_mask_15 = attention_mask.bool()
        invert_14 = ~attention_mask_15
        attention_mask_15 = None
        attention_scores_59 = attention_scores_58.masked_fill(
            invert_14, -3.4028234663852886e38
        )
        attention_scores_58 = invert_14 = None
        attention_probs_28 = torch.nn.functional.softmax(attention_scores_59, dim=-1)
        attention_scores_59 = None
        attention_probs_29 = torch.nn.functional.dropout(
            attention_probs_28, 0.1, False, False
        )
        attention_probs_28 = None
        view_207 = attention_probs_29.view(-1, 11, 11)
        attention_probs_29 = None
        context_layer_42 = torch.bmm(view_207, value_layer_14)
        view_207 = value_layer_14 = None
        view_208 = context_layer_42.view(-1, 16, 11, 64)
        context_layer_42 = None
        permute_89 = view_208.permute(0, 2, 1, 3)
        view_208 = None
        context_layer_43 = permute_89.contiguous()
        permute_89 = None
        context_layer_44 = context_layer_43.view((1, 11, -1))
        context_layer_43 = None
        hidden_states_112 = torch._C._nn.linear(
            context_layer_44,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_44 = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_113 = torch.nn.functional.dropout(
            hidden_states_112, 0.1, False, False
        )
        hidden_states_112 = None
        add_89 = hidden_states_113 + hidden_states_111
        hidden_states_113 = hidden_states_111 = None
        hidden_states_114 = torch.nn.functional.layer_norm(
            add_89,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_89 = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_115 = torch._C._nn.linear(
            hidden_states_114,
            l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_116 = torch._C._nn.gelu(hidden_states_115)
        hidden_states_115 = None
        hidden_states_117 = torch._C._nn.linear(
            hidden_states_116,
            l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_116 = l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_118 = torch.nn.functional.dropout(
            hidden_states_117, 0.1, False, False
        )
        hidden_states_117 = None
        add_90 = hidden_states_118 + hidden_states_114
        hidden_states_118 = hidden_states_114 = None
        hidden_states_119 = torch.nn.functional.layer_norm(
            add_90,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_90 = l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_120 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_75 = linear_120.view((1, 11, 16, -1))
        linear_120 = None
        permute_90 = x_75.permute(0, 2, 1, 3)
        x_75 = None
        contiguous_90 = permute_90.contiguous()
        permute_90 = None
        query_layer_15 = contiguous_90.view(-1, 11, 64)
        contiguous_90 = None
        linear_121 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_76 = linear_121.view((1, 11, 16, -1))
        linear_121 = None
        permute_91 = x_76.permute(0, 2, 1, 3)
        x_76 = None
        contiguous_91 = permute_91.contiguous()
        permute_91 = None
        key_layer_15 = contiguous_91.view(-1, 11, 64)
        contiguous_91 = None
        linear_122 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_77 = linear_122.view((1, 11, 16, -1))
        linear_122 = None
        permute_92 = x_77.permute(0, 2, 1, 3)
        x_77 = None
        contiguous_92 = permute_92.contiguous()
        permute_92 = None
        value_layer_15 = contiguous_92.view(-1, 11, 64)
        contiguous_92 = None
        tensor_47 = torch.tensor(64, dtype=torch.float32)
        mul_49 = tensor_47 * 3
        tensor_47 = None
        scale_45 = torch.sqrt(mul_49)
        mul_49 = None
        transpose_60 = key_layer_15.transpose(-1, -2)
        to_62 = scale_45.to(dtype=torch.float32)
        scale_45 = None
        truediv_47 = transpose_60 / to_62
        transpose_60 = to_62 = None
        attention_scores_60 = torch.bmm(query_layer_15, truediv_47)
        truediv_47 = None
        rel_embeddings_31 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_30 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_31 = relative_pos_30.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_30 = None
        getitem_19 = rel_embeddings_31[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_31 = None
        rel_embeddings_32 = getitem_19.unsqueeze(0)
        getitem_19 = None
        linear_123 = torch._C._nn.linear(
            rel_embeddings_32,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_78 = linear_123.view((1, 512, 16, -1))
        linear_123 = None
        permute_93 = x_78.permute(0, 2, 1, 3)
        x_78 = None
        contiguous_93 = permute_93.contiguous()
        permute_93 = None
        view_217 = contiguous_93.view(-1, 512, 64)
        contiguous_93 = None
        pos_query_layer_15 = view_217.repeat(1, 1, 1)
        view_217 = None
        linear_124 = torch._C._nn.linear(
            rel_embeddings_32,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_32 = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_79 = linear_124.view((1, 512, 16, -1))
        linear_124 = None
        permute_94 = x_79.permute(0, 2, 1, 3)
        x_79 = None
        contiguous_94 = permute_94.contiguous()
        permute_94 = None
        view_219 = contiguous_94.view(-1, 512, 64)
        contiguous_94 = None
        pos_key_layer_15 = view_219.repeat(1, 1, 1)
        view_219 = None
        tensor_48 = torch.tensor(64, dtype=torch.float32)
        mul_50 = tensor_48 * 3
        tensor_48 = None
        scale_46 = torch.sqrt(mul_50)
        mul_50 = None
        transpose_61 = pos_key_layer_15.transpose(-1, -2)
        pos_key_layer_15 = None
        c2p_att_30 = torch.bmm(query_layer_15, transpose_61)
        query_layer_15 = transpose_61 = None
        add_91 = relative_pos_31 + 256
        c2p_pos_15 = torch.clamp(add_91, 0, 511)
        add_91 = None
        squeeze_31 = c2p_pos_15.squeeze(0)
        c2p_pos_15 = None
        expand_30 = squeeze_31.expand([16, 11, 11])
        squeeze_31 = None
        c2p_att_31 = torch.gather(c2p_att_30, dim=-1, index=expand_30)
        c2p_att_30 = expand_30 = None
        to_64 = scale_46.to(dtype=torch.float32)
        scale_46 = None
        truediv_48 = c2p_att_31 / to_64
        c2p_att_31 = to_64 = None
        score_30 = 0 + truediv_48
        truediv_48 = None
        tensor_49 = torch.tensor(64, dtype=torch.float32)
        mul_51 = tensor_49 * 3
        tensor_49 = None
        scale_47 = torch.sqrt(mul_51)
        mul_51 = None
        neg_15 = -relative_pos_31
        relative_pos_31 = None
        add_93 = neg_15 + 256
        neg_15 = None
        p2c_pos_15 = torch.clamp(add_93, 0, 511)
        add_93 = None
        transpose_62 = pos_query_layer_15.transpose(-1, -2)
        pos_query_layer_15 = None
        p2c_att_30 = torch.bmm(key_layer_15, transpose_62)
        key_layer_15 = transpose_62 = None
        squeeze_32 = p2c_pos_15.squeeze(0)
        p2c_pos_15 = None
        expand_31 = squeeze_32.expand([16, 11, 11])
        squeeze_32 = None
        gather_31 = torch.gather(p2c_att_30, dim=-1, index=expand_31)
        p2c_att_30 = expand_31 = None
        p2c_att_31 = gather_31.transpose(-1, -2)
        gather_31 = None
        to_65 = scale_47.to(dtype=torch.float32)
        scale_47 = None
        truediv_49 = p2c_att_31 / to_65
        p2c_att_31 = to_65 = None
        score_30 += truediv_49
        score_31 = score_30
        score_30 = truediv_49 = None
        attention_scores_61 = attention_scores_60 + score_31
        attention_scores_60 = score_31 = None
        attention_scores_62 = attention_scores_61.view(-1, 16, 11, 11)
        attention_scores_61 = None
        attention_mask_16 = attention_mask.bool()
        invert_15 = ~attention_mask_16
        attention_mask_16 = None
        attention_scores_63 = attention_scores_62.masked_fill(
            invert_15, -3.4028234663852886e38
        )
        attention_scores_62 = invert_15 = None
        attention_probs_30 = torch.nn.functional.softmax(attention_scores_63, dim=-1)
        attention_scores_63 = None
        attention_probs_31 = torch.nn.functional.dropout(
            attention_probs_30, 0.1, False, False
        )
        attention_probs_30 = None
        view_221 = attention_probs_31.view(-1, 11, 11)
        attention_probs_31 = None
        context_layer_45 = torch.bmm(view_221, value_layer_15)
        view_221 = value_layer_15 = None
        view_222 = context_layer_45.view(-1, 16, 11, 64)
        context_layer_45 = None
        permute_95 = view_222.permute(0, 2, 1, 3)
        view_222 = None
        context_layer_46 = permute_95.contiguous()
        permute_95 = None
        context_layer_47 = context_layer_46.view((1, 11, -1))
        context_layer_46 = None
        hidden_states_120 = torch._C._nn.linear(
            context_layer_47,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_47 = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_121 = torch.nn.functional.dropout(
            hidden_states_120, 0.1, False, False
        )
        hidden_states_120 = None
        add_95 = hidden_states_121 + hidden_states_119
        hidden_states_121 = hidden_states_119 = None
        hidden_states_122 = torch.nn.functional.layer_norm(
            add_95,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_95 = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_123 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_124 = torch._C._nn.gelu(hidden_states_123)
        hidden_states_123 = None
        hidden_states_125 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_124 = l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_126 = torch.nn.functional.dropout(
            hidden_states_125, 0.1, False, False
        )
        hidden_states_125 = None
        add_96 = hidden_states_126 + hidden_states_122
        hidden_states_126 = hidden_states_122 = None
        hidden_states_127 = torch.nn.functional.layer_norm(
            add_96,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_96 = l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_128 = torch._C._nn.linear(
            hidden_states_127,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_80 = linear_128.view((1, 11, 16, -1))
        linear_128 = None
        permute_96 = x_80.permute(0, 2, 1, 3)
        x_80 = None
        contiguous_96 = permute_96.contiguous()
        permute_96 = None
        query_layer_16 = contiguous_96.view(-1, 11, 64)
        contiguous_96 = None
        linear_129 = torch._C._nn.linear(
            hidden_states_127,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_81 = linear_129.view((1, 11, 16, -1))
        linear_129 = None
        permute_97 = x_81.permute(0, 2, 1, 3)
        x_81 = None
        contiguous_97 = permute_97.contiguous()
        permute_97 = None
        key_layer_16 = contiguous_97.view(-1, 11, 64)
        contiguous_97 = None
        linear_130 = torch._C._nn.linear(
            hidden_states_127,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_82 = linear_130.view((1, 11, 16, -1))
        linear_130 = None
        permute_98 = x_82.permute(0, 2, 1, 3)
        x_82 = None
        contiguous_98 = permute_98.contiguous()
        permute_98 = None
        value_layer_16 = contiguous_98.view(-1, 11, 64)
        contiguous_98 = None
        tensor_50 = torch.tensor(64, dtype=torch.float32)
        mul_52 = tensor_50 * 3
        tensor_50 = None
        scale_48 = torch.sqrt(mul_52)
        mul_52 = None
        transpose_64 = key_layer_16.transpose(-1, -2)
        to_66 = scale_48.to(dtype=torch.float32)
        scale_48 = None
        truediv_50 = transpose_64 / to_66
        transpose_64 = to_66 = None
        attention_scores_64 = torch.bmm(query_layer_16, truediv_50)
        truediv_50 = None
        rel_embeddings_33 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_32 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_33 = relative_pos_32.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_32 = None
        getitem_20 = rel_embeddings_33[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_33 = None
        rel_embeddings_34 = getitem_20.unsqueeze(0)
        getitem_20 = None
        linear_131 = torch._C._nn.linear(
            rel_embeddings_34,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_83 = linear_131.view((1, 512, 16, -1))
        linear_131 = None
        permute_99 = x_83.permute(0, 2, 1, 3)
        x_83 = None
        contiguous_99 = permute_99.contiguous()
        permute_99 = None
        view_231 = contiguous_99.view(-1, 512, 64)
        contiguous_99 = None
        pos_query_layer_16 = view_231.repeat(1, 1, 1)
        view_231 = None
        linear_132 = torch._C._nn.linear(
            rel_embeddings_34,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_34 = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_84 = linear_132.view((1, 512, 16, -1))
        linear_132 = None
        permute_100 = x_84.permute(0, 2, 1, 3)
        x_84 = None
        contiguous_100 = permute_100.contiguous()
        permute_100 = None
        view_233 = contiguous_100.view(-1, 512, 64)
        contiguous_100 = None
        pos_key_layer_16 = view_233.repeat(1, 1, 1)
        view_233 = None
        tensor_51 = torch.tensor(64, dtype=torch.float32)
        mul_53 = tensor_51 * 3
        tensor_51 = None
        scale_49 = torch.sqrt(mul_53)
        mul_53 = None
        transpose_65 = pos_key_layer_16.transpose(-1, -2)
        pos_key_layer_16 = None
        c2p_att_32 = torch.bmm(query_layer_16, transpose_65)
        query_layer_16 = transpose_65 = None
        add_97 = relative_pos_33 + 256
        c2p_pos_16 = torch.clamp(add_97, 0, 511)
        add_97 = None
        squeeze_33 = c2p_pos_16.squeeze(0)
        c2p_pos_16 = None
        expand_32 = squeeze_33.expand([16, 11, 11])
        squeeze_33 = None
        c2p_att_33 = torch.gather(c2p_att_32, dim=-1, index=expand_32)
        c2p_att_32 = expand_32 = None
        to_68 = scale_49.to(dtype=torch.float32)
        scale_49 = None
        truediv_51 = c2p_att_33 / to_68
        c2p_att_33 = to_68 = None
        score_32 = 0 + truediv_51
        truediv_51 = None
        tensor_52 = torch.tensor(64, dtype=torch.float32)
        mul_54 = tensor_52 * 3
        tensor_52 = None
        scale_50 = torch.sqrt(mul_54)
        mul_54 = None
        neg_16 = -relative_pos_33
        relative_pos_33 = None
        add_99 = neg_16 + 256
        neg_16 = None
        p2c_pos_16 = torch.clamp(add_99, 0, 511)
        add_99 = None
        transpose_66 = pos_query_layer_16.transpose(-1, -2)
        pos_query_layer_16 = None
        p2c_att_32 = torch.bmm(key_layer_16, transpose_66)
        key_layer_16 = transpose_66 = None
        squeeze_34 = p2c_pos_16.squeeze(0)
        p2c_pos_16 = None
        expand_33 = squeeze_34.expand([16, 11, 11])
        squeeze_34 = None
        gather_33 = torch.gather(p2c_att_32, dim=-1, index=expand_33)
        p2c_att_32 = expand_33 = None
        p2c_att_33 = gather_33.transpose(-1, -2)
        gather_33 = None
        to_69 = scale_50.to(dtype=torch.float32)
        scale_50 = None
        truediv_52 = p2c_att_33 / to_69
        p2c_att_33 = to_69 = None
        score_32 += truediv_52
        score_33 = score_32
        score_32 = truediv_52 = None
        attention_scores_65 = attention_scores_64 + score_33
        attention_scores_64 = score_33 = None
        attention_scores_66 = attention_scores_65.view(-1, 16, 11, 11)
        attention_scores_65 = None
        attention_mask_17 = attention_mask.bool()
        invert_16 = ~attention_mask_17
        attention_mask_17 = None
        attention_scores_67 = attention_scores_66.masked_fill(
            invert_16, -3.4028234663852886e38
        )
        attention_scores_66 = invert_16 = None
        attention_probs_32 = torch.nn.functional.softmax(attention_scores_67, dim=-1)
        attention_scores_67 = None
        attention_probs_33 = torch.nn.functional.dropout(
            attention_probs_32, 0.1, False, False
        )
        attention_probs_32 = None
        view_235 = attention_probs_33.view(-1, 11, 11)
        attention_probs_33 = None
        context_layer_48 = torch.bmm(view_235, value_layer_16)
        view_235 = value_layer_16 = None
        view_236 = context_layer_48.view(-1, 16, 11, 64)
        context_layer_48 = None
        permute_101 = view_236.permute(0, 2, 1, 3)
        view_236 = None
        context_layer_49 = permute_101.contiguous()
        permute_101 = None
        context_layer_50 = context_layer_49.view((1, 11, -1))
        context_layer_49 = None
        hidden_states_128 = torch._C._nn.linear(
            context_layer_50,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_50 = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_129 = torch.nn.functional.dropout(
            hidden_states_128, 0.1, False, False
        )
        hidden_states_128 = None
        add_101 = hidden_states_129 + hidden_states_127
        hidden_states_129 = hidden_states_127 = None
        hidden_states_130 = torch.nn.functional.layer_norm(
            add_101,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_101 = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_131 = torch._C._nn.linear(
            hidden_states_130,
            l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_132 = torch._C._nn.gelu(hidden_states_131)
        hidden_states_131 = None
        hidden_states_133 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_132 = l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_134 = torch.nn.functional.dropout(
            hidden_states_133, 0.1, False, False
        )
        hidden_states_133 = None
        add_102 = hidden_states_134 + hidden_states_130
        hidden_states_134 = hidden_states_130 = None
        hidden_states_135 = torch.nn.functional.layer_norm(
            add_102,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_102 = l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_136 = torch._C._nn.linear(
            hidden_states_135,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_85 = linear_136.view((1, 11, 16, -1))
        linear_136 = None
        permute_102 = x_85.permute(0, 2, 1, 3)
        x_85 = None
        contiguous_102 = permute_102.contiguous()
        permute_102 = None
        query_layer_17 = contiguous_102.view(-1, 11, 64)
        contiguous_102 = None
        linear_137 = torch._C._nn.linear(
            hidden_states_135,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_86 = linear_137.view((1, 11, 16, -1))
        linear_137 = None
        permute_103 = x_86.permute(0, 2, 1, 3)
        x_86 = None
        contiguous_103 = permute_103.contiguous()
        permute_103 = None
        key_layer_17 = contiguous_103.view(-1, 11, 64)
        contiguous_103 = None
        linear_138 = torch._C._nn.linear(
            hidden_states_135,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_87 = linear_138.view((1, 11, 16, -1))
        linear_138 = None
        permute_104 = x_87.permute(0, 2, 1, 3)
        x_87 = None
        contiguous_104 = permute_104.contiguous()
        permute_104 = None
        value_layer_17 = contiguous_104.view(-1, 11, 64)
        contiguous_104 = None
        tensor_53 = torch.tensor(64, dtype=torch.float32)
        mul_55 = tensor_53 * 3
        tensor_53 = None
        scale_51 = torch.sqrt(mul_55)
        mul_55 = None
        transpose_68 = key_layer_17.transpose(-1, -2)
        to_70 = scale_51.to(dtype=torch.float32)
        scale_51 = None
        truediv_53 = transpose_68 / to_70
        transpose_68 = to_70 = None
        attention_scores_68 = torch.bmm(query_layer_17, truediv_53)
        truediv_53 = None
        rel_embeddings_35 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_34 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_35 = relative_pos_34.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_34 = None
        getitem_21 = rel_embeddings_35[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_35 = None
        rel_embeddings_36 = getitem_21.unsqueeze(0)
        getitem_21 = None
        linear_139 = torch._C._nn.linear(
            rel_embeddings_36,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_88 = linear_139.view((1, 512, 16, -1))
        linear_139 = None
        permute_105 = x_88.permute(0, 2, 1, 3)
        x_88 = None
        contiguous_105 = permute_105.contiguous()
        permute_105 = None
        view_245 = contiguous_105.view(-1, 512, 64)
        contiguous_105 = None
        pos_query_layer_17 = view_245.repeat(1, 1, 1)
        view_245 = None
        linear_140 = torch._C._nn.linear(
            rel_embeddings_36,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_36 = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_89 = linear_140.view((1, 512, 16, -1))
        linear_140 = None
        permute_106 = x_89.permute(0, 2, 1, 3)
        x_89 = None
        contiguous_106 = permute_106.contiguous()
        permute_106 = None
        view_247 = contiguous_106.view(-1, 512, 64)
        contiguous_106 = None
        pos_key_layer_17 = view_247.repeat(1, 1, 1)
        view_247 = None
        tensor_54 = torch.tensor(64, dtype=torch.float32)
        mul_56 = tensor_54 * 3
        tensor_54 = None
        scale_52 = torch.sqrt(mul_56)
        mul_56 = None
        transpose_69 = pos_key_layer_17.transpose(-1, -2)
        pos_key_layer_17 = None
        c2p_att_34 = torch.bmm(query_layer_17, transpose_69)
        query_layer_17 = transpose_69 = None
        add_103 = relative_pos_35 + 256
        c2p_pos_17 = torch.clamp(add_103, 0, 511)
        add_103 = None
        squeeze_35 = c2p_pos_17.squeeze(0)
        c2p_pos_17 = None
        expand_34 = squeeze_35.expand([16, 11, 11])
        squeeze_35 = None
        c2p_att_35 = torch.gather(c2p_att_34, dim=-1, index=expand_34)
        c2p_att_34 = expand_34 = None
        to_72 = scale_52.to(dtype=torch.float32)
        scale_52 = None
        truediv_54 = c2p_att_35 / to_72
        c2p_att_35 = to_72 = None
        score_34 = 0 + truediv_54
        truediv_54 = None
        tensor_55 = torch.tensor(64, dtype=torch.float32)
        mul_57 = tensor_55 * 3
        tensor_55 = None
        scale_53 = torch.sqrt(mul_57)
        mul_57 = None
        neg_17 = -relative_pos_35
        relative_pos_35 = None
        add_105 = neg_17 + 256
        neg_17 = None
        p2c_pos_17 = torch.clamp(add_105, 0, 511)
        add_105 = None
        transpose_70 = pos_query_layer_17.transpose(-1, -2)
        pos_query_layer_17 = None
        p2c_att_34 = torch.bmm(key_layer_17, transpose_70)
        key_layer_17 = transpose_70 = None
        squeeze_36 = p2c_pos_17.squeeze(0)
        p2c_pos_17 = None
        expand_35 = squeeze_36.expand([16, 11, 11])
        squeeze_36 = None
        gather_35 = torch.gather(p2c_att_34, dim=-1, index=expand_35)
        p2c_att_34 = expand_35 = None
        p2c_att_35 = gather_35.transpose(-1, -2)
        gather_35 = None
        to_73 = scale_53.to(dtype=torch.float32)
        scale_53 = None
        truediv_55 = p2c_att_35 / to_73
        p2c_att_35 = to_73 = None
        score_34 += truediv_55
        score_35 = score_34
        score_34 = truediv_55 = None
        attention_scores_69 = attention_scores_68 + score_35
        attention_scores_68 = score_35 = None
        attention_scores_70 = attention_scores_69.view(-1, 16, 11, 11)
        attention_scores_69 = None
        attention_mask_18 = attention_mask.bool()
        invert_17 = ~attention_mask_18
        attention_mask_18 = None
        attention_scores_71 = attention_scores_70.masked_fill(
            invert_17, -3.4028234663852886e38
        )
        attention_scores_70 = invert_17 = None
        attention_probs_34 = torch.nn.functional.softmax(attention_scores_71, dim=-1)
        attention_scores_71 = None
        attention_probs_35 = torch.nn.functional.dropout(
            attention_probs_34, 0.1, False, False
        )
        attention_probs_34 = None
        view_249 = attention_probs_35.view(-1, 11, 11)
        attention_probs_35 = None
        context_layer_51 = torch.bmm(view_249, value_layer_17)
        view_249 = value_layer_17 = None
        view_250 = context_layer_51.view(-1, 16, 11, 64)
        context_layer_51 = None
        permute_107 = view_250.permute(0, 2, 1, 3)
        view_250 = None
        context_layer_52 = permute_107.contiguous()
        permute_107 = None
        context_layer_53 = context_layer_52.view((1, 11, -1))
        context_layer_52 = None
        hidden_states_136 = torch._C._nn.linear(
            context_layer_53,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_53 = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_137 = torch.nn.functional.dropout(
            hidden_states_136, 0.1, False, False
        )
        hidden_states_136 = None
        add_107 = hidden_states_137 + hidden_states_135
        hidden_states_137 = hidden_states_135 = None
        hidden_states_138 = torch.nn.functional.layer_norm(
            add_107,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_107 = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_139 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_140 = torch._C._nn.gelu(hidden_states_139)
        hidden_states_139 = None
        hidden_states_141 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_140 = l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_142 = torch.nn.functional.dropout(
            hidden_states_141, 0.1, False, False
        )
        hidden_states_141 = None
        add_108 = hidden_states_142 + hidden_states_138
        hidden_states_142 = hidden_states_138 = None
        hidden_states_143 = torch.nn.functional.layer_norm(
            add_108,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_108 = l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_144 = torch._C._nn.linear(
            hidden_states_143,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_90 = linear_144.view((1, 11, 16, -1))
        linear_144 = None
        permute_108 = x_90.permute(0, 2, 1, 3)
        x_90 = None
        contiguous_108 = permute_108.contiguous()
        permute_108 = None
        query_layer_18 = contiguous_108.view(-1, 11, 64)
        contiguous_108 = None
        linear_145 = torch._C._nn.linear(
            hidden_states_143,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_91 = linear_145.view((1, 11, 16, -1))
        linear_145 = None
        permute_109 = x_91.permute(0, 2, 1, 3)
        x_91 = None
        contiguous_109 = permute_109.contiguous()
        permute_109 = None
        key_layer_18 = contiguous_109.view(-1, 11, 64)
        contiguous_109 = None
        linear_146 = torch._C._nn.linear(
            hidden_states_143,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_92 = linear_146.view((1, 11, 16, -1))
        linear_146 = None
        permute_110 = x_92.permute(0, 2, 1, 3)
        x_92 = None
        contiguous_110 = permute_110.contiguous()
        permute_110 = None
        value_layer_18 = contiguous_110.view(-1, 11, 64)
        contiguous_110 = None
        tensor_56 = torch.tensor(64, dtype=torch.float32)
        mul_58 = tensor_56 * 3
        tensor_56 = None
        scale_54 = torch.sqrt(mul_58)
        mul_58 = None
        transpose_72 = key_layer_18.transpose(-1, -2)
        to_74 = scale_54.to(dtype=torch.float32)
        scale_54 = None
        truediv_56 = transpose_72 / to_74
        transpose_72 = to_74 = None
        attention_scores_72 = torch.bmm(query_layer_18, truediv_56)
        truediv_56 = None
        rel_embeddings_37 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_36 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_37 = relative_pos_36.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_36 = None
        getitem_22 = rel_embeddings_37[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_37 = None
        rel_embeddings_38 = getitem_22.unsqueeze(0)
        getitem_22 = None
        linear_147 = torch._C._nn.linear(
            rel_embeddings_38,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_93 = linear_147.view((1, 512, 16, -1))
        linear_147 = None
        permute_111 = x_93.permute(0, 2, 1, 3)
        x_93 = None
        contiguous_111 = permute_111.contiguous()
        permute_111 = None
        view_259 = contiguous_111.view(-1, 512, 64)
        contiguous_111 = None
        pos_query_layer_18 = view_259.repeat(1, 1, 1)
        view_259 = None
        linear_148 = torch._C._nn.linear(
            rel_embeddings_38,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_38 = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_94 = linear_148.view((1, 512, 16, -1))
        linear_148 = None
        permute_112 = x_94.permute(0, 2, 1, 3)
        x_94 = None
        contiguous_112 = permute_112.contiguous()
        permute_112 = None
        view_261 = contiguous_112.view(-1, 512, 64)
        contiguous_112 = None
        pos_key_layer_18 = view_261.repeat(1, 1, 1)
        view_261 = None
        tensor_57 = torch.tensor(64, dtype=torch.float32)
        mul_59 = tensor_57 * 3
        tensor_57 = None
        scale_55 = torch.sqrt(mul_59)
        mul_59 = None
        transpose_73 = pos_key_layer_18.transpose(-1, -2)
        pos_key_layer_18 = None
        c2p_att_36 = torch.bmm(query_layer_18, transpose_73)
        query_layer_18 = transpose_73 = None
        add_109 = relative_pos_37 + 256
        c2p_pos_18 = torch.clamp(add_109, 0, 511)
        add_109 = None
        squeeze_37 = c2p_pos_18.squeeze(0)
        c2p_pos_18 = None
        expand_36 = squeeze_37.expand([16, 11, 11])
        squeeze_37 = None
        c2p_att_37 = torch.gather(c2p_att_36, dim=-1, index=expand_36)
        c2p_att_36 = expand_36 = None
        to_76 = scale_55.to(dtype=torch.float32)
        scale_55 = None
        truediv_57 = c2p_att_37 / to_76
        c2p_att_37 = to_76 = None
        score_36 = 0 + truediv_57
        truediv_57 = None
        tensor_58 = torch.tensor(64, dtype=torch.float32)
        mul_60 = tensor_58 * 3
        tensor_58 = None
        scale_56 = torch.sqrt(mul_60)
        mul_60 = None
        neg_18 = -relative_pos_37
        relative_pos_37 = None
        add_111 = neg_18 + 256
        neg_18 = None
        p2c_pos_18 = torch.clamp(add_111, 0, 511)
        add_111 = None
        transpose_74 = pos_query_layer_18.transpose(-1, -2)
        pos_query_layer_18 = None
        p2c_att_36 = torch.bmm(key_layer_18, transpose_74)
        key_layer_18 = transpose_74 = None
        squeeze_38 = p2c_pos_18.squeeze(0)
        p2c_pos_18 = None
        expand_37 = squeeze_38.expand([16, 11, 11])
        squeeze_38 = None
        gather_37 = torch.gather(p2c_att_36, dim=-1, index=expand_37)
        p2c_att_36 = expand_37 = None
        p2c_att_37 = gather_37.transpose(-1, -2)
        gather_37 = None
        to_77 = scale_56.to(dtype=torch.float32)
        scale_56 = None
        truediv_58 = p2c_att_37 / to_77
        p2c_att_37 = to_77 = None
        score_36 += truediv_58
        score_37 = score_36
        score_36 = truediv_58 = None
        attention_scores_73 = attention_scores_72 + score_37
        attention_scores_72 = score_37 = None
        attention_scores_74 = attention_scores_73.view(-1, 16, 11, 11)
        attention_scores_73 = None
        attention_mask_19 = attention_mask.bool()
        invert_18 = ~attention_mask_19
        attention_mask_19 = None
        attention_scores_75 = attention_scores_74.masked_fill(
            invert_18, -3.4028234663852886e38
        )
        attention_scores_74 = invert_18 = None
        attention_probs_36 = torch.nn.functional.softmax(attention_scores_75, dim=-1)
        attention_scores_75 = None
        attention_probs_37 = torch.nn.functional.dropout(
            attention_probs_36, 0.1, False, False
        )
        attention_probs_36 = None
        view_263 = attention_probs_37.view(-1, 11, 11)
        attention_probs_37 = None
        context_layer_54 = torch.bmm(view_263, value_layer_18)
        view_263 = value_layer_18 = None
        view_264 = context_layer_54.view(-1, 16, 11, 64)
        context_layer_54 = None
        permute_113 = view_264.permute(0, 2, 1, 3)
        view_264 = None
        context_layer_55 = permute_113.contiguous()
        permute_113 = None
        context_layer_56 = context_layer_55.view((1, 11, -1))
        context_layer_55 = None
        hidden_states_144 = torch._C._nn.linear(
            context_layer_56,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_56 = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_145 = torch.nn.functional.dropout(
            hidden_states_144, 0.1, False, False
        )
        hidden_states_144 = None
        add_113 = hidden_states_145 + hidden_states_143
        hidden_states_145 = hidden_states_143 = None
        hidden_states_146 = torch.nn.functional.layer_norm(
            add_113,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_113 = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_147 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_148 = torch._C._nn.gelu(hidden_states_147)
        hidden_states_147 = None
        hidden_states_149 = torch._C._nn.linear(
            hidden_states_148,
            l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_148 = l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_150 = torch.nn.functional.dropout(
            hidden_states_149, 0.1, False, False
        )
        hidden_states_149 = None
        add_114 = hidden_states_150 + hidden_states_146
        hidden_states_150 = hidden_states_146 = None
        hidden_states_151 = torch.nn.functional.layer_norm(
            add_114,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_114 = l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_152 = torch._C._nn.linear(
            hidden_states_151,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_95 = linear_152.view((1, 11, 16, -1))
        linear_152 = None
        permute_114 = x_95.permute(0, 2, 1, 3)
        x_95 = None
        contiguous_114 = permute_114.contiguous()
        permute_114 = None
        query_layer_19 = contiguous_114.view(-1, 11, 64)
        contiguous_114 = None
        linear_153 = torch._C._nn.linear(
            hidden_states_151,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_96 = linear_153.view((1, 11, 16, -1))
        linear_153 = None
        permute_115 = x_96.permute(0, 2, 1, 3)
        x_96 = None
        contiguous_115 = permute_115.contiguous()
        permute_115 = None
        key_layer_19 = contiguous_115.view(-1, 11, 64)
        contiguous_115 = None
        linear_154 = torch._C._nn.linear(
            hidden_states_151,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_97 = linear_154.view((1, 11, 16, -1))
        linear_154 = None
        permute_116 = x_97.permute(0, 2, 1, 3)
        x_97 = None
        contiguous_116 = permute_116.contiguous()
        permute_116 = None
        value_layer_19 = contiguous_116.view(-1, 11, 64)
        contiguous_116 = None
        tensor_59 = torch.tensor(64, dtype=torch.float32)
        mul_61 = tensor_59 * 3
        tensor_59 = None
        scale_57 = torch.sqrt(mul_61)
        mul_61 = None
        transpose_76 = key_layer_19.transpose(-1, -2)
        to_78 = scale_57.to(dtype=torch.float32)
        scale_57 = None
        truediv_59 = transpose_76 / to_78
        transpose_76 = to_78 = None
        attention_scores_76 = torch.bmm(query_layer_19, truediv_59)
        truediv_59 = None
        rel_embeddings_39 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_38 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_39 = relative_pos_38.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_38 = None
        getitem_23 = rel_embeddings_39[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_39 = None
        rel_embeddings_40 = getitem_23.unsqueeze(0)
        getitem_23 = None
        linear_155 = torch._C._nn.linear(
            rel_embeddings_40,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_98 = linear_155.view((1, 512, 16, -1))
        linear_155 = None
        permute_117 = x_98.permute(0, 2, 1, 3)
        x_98 = None
        contiguous_117 = permute_117.contiguous()
        permute_117 = None
        view_273 = contiguous_117.view(-1, 512, 64)
        contiguous_117 = None
        pos_query_layer_19 = view_273.repeat(1, 1, 1)
        view_273 = None
        linear_156 = torch._C._nn.linear(
            rel_embeddings_40,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_40 = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_99 = linear_156.view((1, 512, 16, -1))
        linear_156 = None
        permute_118 = x_99.permute(0, 2, 1, 3)
        x_99 = None
        contiguous_118 = permute_118.contiguous()
        permute_118 = None
        view_275 = contiguous_118.view(-1, 512, 64)
        contiguous_118 = None
        pos_key_layer_19 = view_275.repeat(1, 1, 1)
        view_275 = None
        tensor_60 = torch.tensor(64, dtype=torch.float32)
        mul_62 = tensor_60 * 3
        tensor_60 = None
        scale_58 = torch.sqrt(mul_62)
        mul_62 = None
        transpose_77 = pos_key_layer_19.transpose(-1, -2)
        pos_key_layer_19 = None
        c2p_att_38 = torch.bmm(query_layer_19, transpose_77)
        query_layer_19 = transpose_77 = None
        add_115 = relative_pos_39 + 256
        c2p_pos_19 = torch.clamp(add_115, 0, 511)
        add_115 = None
        squeeze_39 = c2p_pos_19.squeeze(0)
        c2p_pos_19 = None
        expand_38 = squeeze_39.expand([16, 11, 11])
        squeeze_39 = None
        c2p_att_39 = torch.gather(c2p_att_38, dim=-1, index=expand_38)
        c2p_att_38 = expand_38 = None
        to_80 = scale_58.to(dtype=torch.float32)
        scale_58 = None
        truediv_60 = c2p_att_39 / to_80
        c2p_att_39 = to_80 = None
        score_38 = 0 + truediv_60
        truediv_60 = None
        tensor_61 = torch.tensor(64, dtype=torch.float32)
        mul_63 = tensor_61 * 3
        tensor_61 = None
        scale_59 = torch.sqrt(mul_63)
        mul_63 = None
        neg_19 = -relative_pos_39
        relative_pos_39 = None
        add_117 = neg_19 + 256
        neg_19 = None
        p2c_pos_19 = torch.clamp(add_117, 0, 511)
        add_117 = None
        transpose_78 = pos_query_layer_19.transpose(-1, -2)
        pos_query_layer_19 = None
        p2c_att_38 = torch.bmm(key_layer_19, transpose_78)
        key_layer_19 = transpose_78 = None
        squeeze_40 = p2c_pos_19.squeeze(0)
        p2c_pos_19 = None
        expand_39 = squeeze_40.expand([16, 11, 11])
        squeeze_40 = None
        gather_39 = torch.gather(p2c_att_38, dim=-1, index=expand_39)
        p2c_att_38 = expand_39 = None
        p2c_att_39 = gather_39.transpose(-1, -2)
        gather_39 = None
        to_81 = scale_59.to(dtype=torch.float32)
        scale_59 = None
        truediv_61 = p2c_att_39 / to_81
        p2c_att_39 = to_81 = None
        score_38 += truediv_61
        score_39 = score_38
        score_38 = truediv_61 = None
        attention_scores_77 = attention_scores_76 + score_39
        attention_scores_76 = score_39 = None
        attention_scores_78 = attention_scores_77.view(-1, 16, 11, 11)
        attention_scores_77 = None
        attention_mask_20 = attention_mask.bool()
        invert_19 = ~attention_mask_20
        attention_mask_20 = None
        attention_scores_79 = attention_scores_78.masked_fill(
            invert_19, -3.4028234663852886e38
        )
        attention_scores_78 = invert_19 = None
        attention_probs_38 = torch.nn.functional.softmax(attention_scores_79, dim=-1)
        attention_scores_79 = None
        attention_probs_39 = torch.nn.functional.dropout(
            attention_probs_38, 0.1, False, False
        )
        attention_probs_38 = None
        view_277 = attention_probs_39.view(-1, 11, 11)
        attention_probs_39 = None
        context_layer_57 = torch.bmm(view_277, value_layer_19)
        view_277 = value_layer_19 = None
        view_278 = context_layer_57.view(-1, 16, 11, 64)
        context_layer_57 = None
        permute_119 = view_278.permute(0, 2, 1, 3)
        view_278 = None
        context_layer_58 = permute_119.contiguous()
        permute_119 = None
        context_layer_59 = context_layer_58.view((1, 11, -1))
        context_layer_58 = None
        hidden_states_152 = torch._C._nn.linear(
            context_layer_59,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_59 = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_153 = torch.nn.functional.dropout(
            hidden_states_152, 0.1, False, False
        )
        hidden_states_152 = None
        add_119 = hidden_states_153 + hidden_states_151
        hidden_states_153 = hidden_states_151 = None
        hidden_states_154 = torch.nn.functional.layer_norm(
            add_119,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_119 = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_155 = torch._C._nn.linear(
            hidden_states_154,
            l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_156 = torch._C._nn.gelu(hidden_states_155)
        hidden_states_155 = None
        hidden_states_157 = torch._C._nn.linear(
            hidden_states_156,
            l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_156 = l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_158 = torch.nn.functional.dropout(
            hidden_states_157, 0.1, False, False
        )
        hidden_states_157 = None
        add_120 = hidden_states_158 + hidden_states_154
        hidden_states_158 = hidden_states_154 = None
        hidden_states_159 = torch.nn.functional.layer_norm(
            add_120,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_120 = l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_160 = torch._C._nn.linear(
            hidden_states_159,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_100 = linear_160.view((1, 11, 16, -1))
        linear_160 = None
        permute_120 = x_100.permute(0, 2, 1, 3)
        x_100 = None
        contiguous_120 = permute_120.contiguous()
        permute_120 = None
        query_layer_20 = contiguous_120.view(-1, 11, 64)
        contiguous_120 = None
        linear_161 = torch._C._nn.linear(
            hidden_states_159,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_101 = linear_161.view((1, 11, 16, -1))
        linear_161 = None
        permute_121 = x_101.permute(0, 2, 1, 3)
        x_101 = None
        contiguous_121 = permute_121.contiguous()
        permute_121 = None
        key_layer_20 = contiguous_121.view(-1, 11, 64)
        contiguous_121 = None
        linear_162 = torch._C._nn.linear(
            hidden_states_159,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_102 = linear_162.view((1, 11, 16, -1))
        linear_162 = None
        permute_122 = x_102.permute(0, 2, 1, 3)
        x_102 = None
        contiguous_122 = permute_122.contiguous()
        permute_122 = None
        value_layer_20 = contiguous_122.view(-1, 11, 64)
        contiguous_122 = None
        tensor_62 = torch.tensor(64, dtype=torch.float32)
        mul_64 = tensor_62 * 3
        tensor_62 = None
        scale_60 = torch.sqrt(mul_64)
        mul_64 = None
        transpose_80 = key_layer_20.transpose(-1, -2)
        to_82 = scale_60.to(dtype=torch.float32)
        scale_60 = None
        truediv_62 = transpose_80 / to_82
        transpose_80 = to_82 = None
        attention_scores_80 = torch.bmm(query_layer_20, truediv_62)
        truediv_62 = None
        rel_embeddings_41 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_40 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_41 = relative_pos_40.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_40 = None
        getitem_24 = rel_embeddings_41[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_41 = None
        rel_embeddings_42 = getitem_24.unsqueeze(0)
        getitem_24 = None
        linear_163 = torch._C._nn.linear(
            rel_embeddings_42,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_103 = linear_163.view((1, 512, 16, -1))
        linear_163 = None
        permute_123 = x_103.permute(0, 2, 1, 3)
        x_103 = None
        contiguous_123 = permute_123.contiguous()
        permute_123 = None
        view_287 = contiguous_123.view(-1, 512, 64)
        contiguous_123 = None
        pos_query_layer_20 = view_287.repeat(1, 1, 1)
        view_287 = None
        linear_164 = torch._C._nn.linear(
            rel_embeddings_42,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_42 = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_104 = linear_164.view((1, 512, 16, -1))
        linear_164 = None
        permute_124 = x_104.permute(0, 2, 1, 3)
        x_104 = None
        contiguous_124 = permute_124.contiguous()
        permute_124 = None
        view_289 = contiguous_124.view(-1, 512, 64)
        contiguous_124 = None
        pos_key_layer_20 = view_289.repeat(1, 1, 1)
        view_289 = None
        tensor_63 = torch.tensor(64, dtype=torch.float32)
        mul_65 = tensor_63 * 3
        tensor_63 = None
        scale_61 = torch.sqrt(mul_65)
        mul_65 = None
        transpose_81 = pos_key_layer_20.transpose(-1, -2)
        pos_key_layer_20 = None
        c2p_att_40 = torch.bmm(query_layer_20, transpose_81)
        query_layer_20 = transpose_81 = None
        add_121 = relative_pos_41 + 256
        c2p_pos_20 = torch.clamp(add_121, 0, 511)
        add_121 = None
        squeeze_41 = c2p_pos_20.squeeze(0)
        c2p_pos_20 = None
        expand_40 = squeeze_41.expand([16, 11, 11])
        squeeze_41 = None
        c2p_att_41 = torch.gather(c2p_att_40, dim=-1, index=expand_40)
        c2p_att_40 = expand_40 = None
        to_84 = scale_61.to(dtype=torch.float32)
        scale_61 = None
        truediv_63 = c2p_att_41 / to_84
        c2p_att_41 = to_84 = None
        score_40 = 0 + truediv_63
        truediv_63 = None
        tensor_64 = torch.tensor(64, dtype=torch.float32)
        mul_66 = tensor_64 * 3
        tensor_64 = None
        scale_62 = torch.sqrt(mul_66)
        mul_66 = None
        neg_20 = -relative_pos_41
        relative_pos_41 = None
        add_123 = neg_20 + 256
        neg_20 = None
        p2c_pos_20 = torch.clamp(add_123, 0, 511)
        add_123 = None
        transpose_82 = pos_query_layer_20.transpose(-1, -2)
        pos_query_layer_20 = None
        p2c_att_40 = torch.bmm(key_layer_20, transpose_82)
        key_layer_20 = transpose_82 = None
        squeeze_42 = p2c_pos_20.squeeze(0)
        p2c_pos_20 = None
        expand_41 = squeeze_42.expand([16, 11, 11])
        squeeze_42 = None
        gather_41 = torch.gather(p2c_att_40, dim=-1, index=expand_41)
        p2c_att_40 = expand_41 = None
        p2c_att_41 = gather_41.transpose(-1, -2)
        gather_41 = None
        to_85 = scale_62.to(dtype=torch.float32)
        scale_62 = None
        truediv_64 = p2c_att_41 / to_85
        p2c_att_41 = to_85 = None
        score_40 += truediv_64
        score_41 = score_40
        score_40 = truediv_64 = None
        attention_scores_81 = attention_scores_80 + score_41
        attention_scores_80 = score_41 = None
        attention_scores_82 = attention_scores_81.view(-1, 16, 11, 11)
        attention_scores_81 = None
        attention_mask_21 = attention_mask.bool()
        invert_20 = ~attention_mask_21
        attention_mask_21 = None
        attention_scores_83 = attention_scores_82.masked_fill(
            invert_20, -3.4028234663852886e38
        )
        attention_scores_82 = invert_20 = None
        attention_probs_40 = torch.nn.functional.softmax(attention_scores_83, dim=-1)
        attention_scores_83 = None
        attention_probs_41 = torch.nn.functional.dropout(
            attention_probs_40, 0.1, False, False
        )
        attention_probs_40 = None
        view_291 = attention_probs_41.view(-1, 11, 11)
        attention_probs_41 = None
        context_layer_60 = torch.bmm(view_291, value_layer_20)
        view_291 = value_layer_20 = None
        view_292 = context_layer_60.view(-1, 16, 11, 64)
        context_layer_60 = None
        permute_125 = view_292.permute(0, 2, 1, 3)
        view_292 = None
        context_layer_61 = permute_125.contiguous()
        permute_125 = None
        context_layer_62 = context_layer_61.view((1, 11, -1))
        context_layer_61 = None
        hidden_states_160 = torch._C._nn.linear(
            context_layer_62,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_62 = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_161 = torch.nn.functional.dropout(
            hidden_states_160, 0.1, False, False
        )
        hidden_states_160 = None
        add_125 = hidden_states_161 + hidden_states_159
        hidden_states_161 = hidden_states_159 = None
        hidden_states_162 = torch.nn.functional.layer_norm(
            add_125,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_125 = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_163 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_164 = torch._C._nn.gelu(hidden_states_163)
        hidden_states_163 = None
        hidden_states_165 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_164 = l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_166 = torch.nn.functional.dropout(
            hidden_states_165, 0.1, False, False
        )
        hidden_states_165 = None
        add_126 = hidden_states_166 + hidden_states_162
        hidden_states_166 = hidden_states_162 = None
        hidden_states_167 = torch.nn.functional.layer_norm(
            add_126,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_126 = l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_168 = torch._C._nn.linear(
            hidden_states_167,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_105 = linear_168.view((1, 11, 16, -1))
        linear_168 = None
        permute_126 = x_105.permute(0, 2, 1, 3)
        x_105 = None
        contiguous_126 = permute_126.contiguous()
        permute_126 = None
        query_layer_21 = contiguous_126.view(-1, 11, 64)
        contiguous_126 = None
        linear_169 = torch._C._nn.linear(
            hidden_states_167,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_106 = linear_169.view((1, 11, 16, -1))
        linear_169 = None
        permute_127 = x_106.permute(0, 2, 1, 3)
        x_106 = None
        contiguous_127 = permute_127.contiguous()
        permute_127 = None
        key_layer_21 = contiguous_127.view(-1, 11, 64)
        contiguous_127 = None
        linear_170 = torch._C._nn.linear(
            hidden_states_167,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_107 = linear_170.view((1, 11, 16, -1))
        linear_170 = None
        permute_128 = x_107.permute(0, 2, 1, 3)
        x_107 = None
        contiguous_128 = permute_128.contiguous()
        permute_128 = None
        value_layer_21 = contiguous_128.view(-1, 11, 64)
        contiguous_128 = None
        tensor_65 = torch.tensor(64, dtype=torch.float32)
        mul_67 = tensor_65 * 3
        tensor_65 = None
        scale_63 = torch.sqrt(mul_67)
        mul_67 = None
        transpose_84 = key_layer_21.transpose(-1, -2)
        to_86 = scale_63.to(dtype=torch.float32)
        scale_63 = None
        truediv_65 = transpose_84 / to_86
        transpose_84 = to_86 = None
        attention_scores_84 = torch.bmm(query_layer_21, truediv_65)
        truediv_65 = None
        rel_embeddings_43 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_42 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_43 = relative_pos_42.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_42 = None
        getitem_25 = rel_embeddings_43[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_43 = None
        rel_embeddings_44 = getitem_25.unsqueeze(0)
        getitem_25 = None
        linear_171 = torch._C._nn.linear(
            rel_embeddings_44,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_108 = linear_171.view((1, 512, 16, -1))
        linear_171 = None
        permute_129 = x_108.permute(0, 2, 1, 3)
        x_108 = None
        contiguous_129 = permute_129.contiguous()
        permute_129 = None
        view_301 = contiguous_129.view(-1, 512, 64)
        contiguous_129 = None
        pos_query_layer_21 = view_301.repeat(1, 1, 1)
        view_301 = None
        linear_172 = torch._C._nn.linear(
            rel_embeddings_44,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_44 = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_109 = linear_172.view((1, 512, 16, -1))
        linear_172 = None
        permute_130 = x_109.permute(0, 2, 1, 3)
        x_109 = None
        contiguous_130 = permute_130.contiguous()
        permute_130 = None
        view_303 = contiguous_130.view(-1, 512, 64)
        contiguous_130 = None
        pos_key_layer_21 = view_303.repeat(1, 1, 1)
        view_303 = None
        tensor_66 = torch.tensor(64, dtype=torch.float32)
        mul_68 = tensor_66 * 3
        tensor_66 = None
        scale_64 = torch.sqrt(mul_68)
        mul_68 = None
        transpose_85 = pos_key_layer_21.transpose(-1, -2)
        pos_key_layer_21 = None
        c2p_att_42 = torch.bmm(query_layer_21, transpose_85)
        query_layer_21 = transpose_85 = None
        add_127 = relative_pos_43 + 256
        c2p_pos_21 = torch.clamp(add_127, 0, 511)
        add_127 = None
        squeeze_43 = c2p_pos_21.squeeze(0)
        c2p_pos_21 = None
        expand_42 = squeeze_43.expand([16, 11, 11])
        squeeze_43 = None
        c2p_att_43 = torch.gather(c2p_att_42, dim=-1, index=expand_42)
        c2p_att_42 = expand_42 = None
        to_88 = scale_64.to(dtype=torch.float32)
        scale_64 = None
        truediv_66 = c2p_att_43 / to_88
        c2p_att_43 = to_88 = None
        score_42 = 0 + truediv_66
        truediv_66 = None
        tensor_67 = torch.tensor(64, dtype=torch.float32)
        mul_69 = tensor_67 * 3
        tensor_67 = None
        scale_65 = torch.sqrt(mul_69)
        mul_69 = None
        neg_21 = -relative_pos_43
        relative_pos_43 = None
        add_129 = neg_21 + 256
        neg_21 = None
        p2c_pos_21 = torch.clamp(add_129, 0, 511)
        add_129 = None
        transpose_86 = pos_query_layer_21.transpose(-1, -2)
        pos_query_layer_21 = None
        p2c_att_42 = torch.bmm(key_layer_21, transpose_86)
        key_layer_21 = transpose_86 = None
        squeeze_44 = p2c_pos_21.squeeze(0)
        p2c_pos_21 = None
        expand_43 = squeeze_44.expand([16, 11, 11])
        squeeze_44 = None
        gather_43 = torch.gather(p2c_att_42, dim=-1, index=expand_43)
        p2c_att_42 = expand_43 = None
        p2c_att_43 = gather_43.transpose(-1, -2)
        gather_43 = None
        to_89 = scale_65.to(dtype=torch.float32)
        scale_65 = None
        truediv_67 = p2c_att_43 / to_89
        p2c_att_43 = to_89 = None
        score_42 += truediv_67
        score_43 = score_42
        score_42 = truediv_67 = None
        attention_scores_85 = attention_scores_84 + score_43
        attention_scores_84 = score_43 = None
        attention_scores_86 = attention_scores_85.view(-1, 16, 11, 11)
        attention_scores_85 = None
        attention_mask_22 = attention_mask.bool()
        invert_21 = ~attention_mask_22
        attention_mask_22 = None
        attention_scores_87 = attention_scores_86.masked_fill(
            invert_21, -3.4028234663852886e38
        )
        attention_scores_86 = invert_21 = None
        attention_probs_42 = torch.nn.functional.softmax(attention_scores_87, dim=-1)
        attention_scores_87 = None
        attention_probs_43 = torch.nn.functional.dropout(
            attention_probs_42, 0.1, False, False
        )
        attention_probs_42 = None
        view_305 = attention_probs_43.view(-1, 11, 11)
        attention_probs_43 = None
        context_layer_63 = torch.bmm(view_305, value_layer_21)
        view_305 = value_layer_21 = None
        view_306 = context_layer_63.view(-1, 16, 11, 64)
        context_layer_63 = None
        permute_131 = view_306.permute(0, 2, 1, 3)
        view_306 = None
        context_layer_64 = permute_131.contiguous()
        permute_131 = None
        context_layer_65 = context_layer_64.view((1, 11, -1))
        context_layer_64 = None
        hidden_states_168 = torch._C._nn.linear(
            context_layer_65,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_65 = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_169 = torch.nn.functional.dropout(
            hidden_states_168, 0.1, False, False
        )
        hidden_states_168 = None
        add_131 = hidden_states_169 + hidden_states_167
        hidden_states_169 = hidden_states_167 = None
        hidden_states_170 = torch.nn.functional.layer_norm(
            add_131,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_131 = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_171 = torch._C._nn.linear(
            hidden_states_170,
            l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_172 = torch._C._nn.gelu(hidden_states_171)
        hidden_states_171 = None
        hidden_states_173 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_172 = l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_174 = torch.nn.functional.dropout(
            hidden_states_173, 0.1, False, False
        )
        hidden_states_173 = None
        add_132 = hidden_states_174 + hidden_states_170
        hidden_states_174 = hidden_states_170 = None
        hidden_states_175 = torch.nn.functional.layer_norm(
            add_132,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_132 = l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_176 = torch._C._nn.linear(
            hidden_states_175,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_110 = linear_176.view((1, 11, 16, -1))
        linear_176 = None
        permute_132 = x_110.permute(0, 2, 1, 3)
        x_110 = None
        contiguous_132 = permute_132.contiguous()
        permute_132 = None
        query_layer_22 = contiguous_132.view(-1, 11, 64)
        contiguous_132 = None
        linear_177 = torch._C._nn.linear(
            hidden_states_175,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_111 = linear_177.view((1, 11, 16, -1))
        linear_177 = None
        permute_133 = x_111.permute(0, 2, 1, 3)
        x_111 = None
        contiguous_133 = permute_133.contiguous()
        permute_133 = None
        key_layer_22 = contiguous_133.view(-1, 11, 64)
        contiguous_133 = None
        linear_178 = torch._C._nn.linear(
            hidden_states_175,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_112 = linear_178.view((1, 11, 16, -1))
        linear_178 = None
        permute_134 = x_112.permute(0, 2, 1, 3)
        x_112 = None
        contiguous_134 = permute_134.contiguous()
        permute_134 = None
        value_layer_22 = contiguous_134.view(-1, 11, 64)
        contiguous_134 = None
        tensor_68 = torch.tensor(64, dtype=torch.float32)
        mul_70 = tensor_68 * 3
        tensor_68 = None
        scale_66 = torch.sqrt(mul_70)
        mul_70 = None
        transpose_88 = key_layer_22.transpose(-1, -2)
        to_90 = scale_66.to(dtype=torch.float32)
        scale_66 = None
        truediv_68 = transpose_88 / to_90
        transpose_88 = to_90 = None
        attention_scores_88 = torch.bmm(query_layer_22, truediv_68)
        truediv_68 = None
        rel_embeddings_45 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        relative_pos_44 = rel_pos_ids_3.unsqueeze(1)
        relative_pos_45 = relative_pos_44.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_44 = None
        getitem_26 = rel_embeddings_45[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_45 = None
        rel_embeddings_46 = getitem_26.unsqueeze(0)
        getitem_26 = None
        linear_179 = torch._C._nn.linear(
            rel_embeddings_46,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_113 = linear_179.view((1, 512, 16, -1))
        linear_179 = None
        permute_135 = x_113.permute(0, 2, 1, 3)
        x_113 = None
        contiguous_135 = permute_135.contiguous()
        permute_135 = None
        view_315 = contiguous_135.view(-1, 512, 64)
        contiguous_135 = None
        pos_query_layer_22 = view_315.repeat(1, 1, 1)
        view_315 = None
        linear_180 = torch._C._nn.linear(
            rel_embeddings_46,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_46 = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_114 = linear_180.view((1, 512, 16, -1))
        linear_180 = None
        permute_136 = x_114.permute(0, 2, 1, 3)
        x_114 = None
        contiguous_136 = permute_136.contiguous()
        permute_136 = None
        view_317 = contiguous_136.view(-1, 512, 64)
        contiguous_136 = None
        pos_key_layer_22 = view_317.repeat(1, 1, 1)
        view_317 = None
        tensor_69 = torch.tensor(64, dtype=torch.float32)
        mul_71 = tensor_69 * 3
        tensor_69 = None
        scale_67 = torch.sqrt(mul_71)
        mul_71 = None
        transpose_89 = pos_key_layer_22.transpose(-1, -2)
        pos_key_layer_22 = None
        c2p_att_44 = torch.bmm(query_layer_22, transpose_89)
        query_layer_22 = transpose_89 = None
        add_133 = relative_pos_45 + 256
        c2p_pos_22 = torch.clamp(add_133, 0, 511)
        add_133 = None
        squeeze_45 = c2p_pos_22.squeeze(0)
        c2p_pos_22 = None
        expand_44 = squeeze_45.expand([16, 11, 11])
        squeeze_45 = None
        c2p_att_45 = torch.gather(c2p_att_44, dim=-1, index=expand_44)
        c2p_att_44 = expand_44 = None
        to_92 = scale_67.to(dtype=torch.float32)
        scale_67 = None
        truediv_69 = c2p_att_45 / to_92
        c2p_att_45 = to_92 = None
        score_44 = 0 + truediv_69
        truediv_69 = None
        tensor_70 = torch.tensor(64, dtype=torch.float32)
        mul_72 = tensor_70 * 3
        tensor_70 = None
        scale_68 = torch.sqrt(mul_72)
        mul_72 = None
        neg_22 = -relative_pos_45
        relative_pos_45 = None
        add_135 = neg_22 + 256
        neg_22 = None
        p2c_pos_22 = torch.clamp(add_135, 0, 511)
        add_135 = None
        transpose_90 = pos_query_layer_22.transpose(-1, -2)
        pos_query_layer_22 = None
        p2c_att_44 = torch.bmm(key_layer_22, transpose_90)
        key_layer_22 = transpose_90 = None
        squeeze_46 = p2c_pos_22.squeeze(0)
        p2c_pos_22 = None
        expand_45 = squeeze_46.expand([16, 11, 11])
        squeeze_46 = None
        gather_45 = torch.gather(p2c_att_44, dim=-1, index=expand_45)
        p2c_att_44 = expand_45 = None
        p2c_att_45 = gather_45.transpose(-1, -2)
        gather_45 = None
        to_93 = scale_68.to(dtype=torch.float32)
        scale_68 = None
        truediv_70 = p2c_att_45 / to_93
        p2c_att_45 = to_93 = None
        score_44 += truediv_70
        score_45 = score_44
        score_44 = truediv_70 = None
        attention_scores_89 = attention_scores_88 + score_45
        attention_scores_88 = score_45 = None
        attention_scores_90 = attention_scores_89.view(-1, 16, 11, 11)
        attention_scores_89 = None
        attention_mask_23 = attention_mask.bool()
        invert_22 = ~attention_mask_23
        attention_mask_23 = None
        attention_scores_91 = attention_scores_90.masked_fill(
            invert_22, -3.4028234663852886e38
        )
        attention_scores_90 = invert_22 = None
        attention_probs_44 = torch.nn.functional.softmax(attention_scores_91, dim=-1)
        attention_scores_91 = None
        attention_probs_45 = torch.nn.functional.dropout(
            attention_probs_44, 0.1, False, False
        )
        attention_probs_44 = None
        view_319 = attention_probs_45.view(-1, 11, 11)
        attention_probs_45 = None
        context_layer_66 = torch.bmm(view_319, value_layer_22)
        view_319 = value_layer_22 = None
        view_320 = context_layer_66.view(-1, 16, 11, 64)
        context_layer_66 = None
        permute_137 = view_320.permute(0, 2, 1, 3)
        view_320 = None
        context_layer_67 = permute_137.contiguous()
        permute_137 = None
        context_layer_68 = context_layer_67.view((1, 11, -1))
        context_layer_67 = None
        hidden_states_176 = torch._C._nn.linear(
            context_layer_68,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_68 = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_177 = torch.nn.functional.dropout(
            hidden_states_176, 0.1, False, False
        )
        hidden_states_176 = None
        add_137 = hidden_states_177 + hidden_states_175
        hidden_states_177 = hidden_states_175 = None
        hidden_states_178 = torch.nn.functional.layer_norm(
            add_137,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_137 = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_179 = torch._C._nn.linear(
            hidden_states_178,
            l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_180 = torch._C._nn.gelu(hidden_states_179)
        hidden_states_179 = None
        hidden_states_181 = torch._C._nn.linear(
            hidden_states_180,
            l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_180 = l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_182 = torch.nn.functional.dropout(
            hidden_states_181, 0.1, False, False
        )
        hidden_states_181 = None
        add_138 = hidden_states_182 + hidden_states_178
        hidden_states_182 = hidden_states_178 = None
        hidden_states_183 = torch.nn.functional.layer_norm(
            add_138,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_138 = l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_184 = torch._C._nn.linear(
            hidden_states_183,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        x_115 = linear_184.view((1, 11, 16, -1))
        linear_184 = None
        permute_138 = x_115.permute(0, 2, 1, 3)
        x_115 = None
        contiguous_138 = permute_138.contiguous()
        permute_138 = None
        query_layer_23 = contiguous_138.view(-1, 11, 64)
        contiguous_138 = None
        linear_185 = torch._C._nn.linear(
            hidden_states_183,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        x_116 = linear_185.view((1, 11, 16, -1))
        linear_185 = None
        permute_139 = x_116.permute(0, 2, 1, 3)
        x_116 = None
        contiguous_139 = permute_139.contiguous()
        permute_139 = None
        key_layer_23 = contiguous_139.view(-1, 11, 64)
        contiguous_139 = None
        linear_186 = torch._C._nn.linear(
            hidden_states_183,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_proj_parameters_bias_ = (None)
        x_117 = linear_186.view((1, 11, 16, -1))
        linear_186 = None
        permute_140 = x_117.permute(0, 2, 1, 3)
        x_117 = None
        contiguous_140 = permute_140.contiguous()
        permute_140 = None
        value_layer_23 = contiguous_140.view(-1, 11, 64)
        contiguous_140 = None
        tensor_71 = torch.tensor(64, dtype=torch.float32)
        mul_73 = tensor_71 * 3
        tensor_71 = None
        scale_69 = torch.sqrt(mul_73)
        mul_73 = None
        transpose_92 = key_layer_23.transpose(-1, -2)
        to_94 = scale_69.to(dtype=torch.float32)
        scale_69 = None
        truediv_71 = transpose_92 / to_94
        transpose_92 = to_94 = None
        attention_scores_92 = torch.bmm(query_layer_23, truediv_71)
        truediv_71 = None
        rel_embeddings_47 = torch.nn.functional.dropout(
            rel_embeddings, 0.1, False, False
        )
        rel_embeddings = None
        relative_pos_46 = rel_pos_ids_3.unsqueeze(1)
        rel_pos_ids_3 = None
        relative_pos_47 = relative_pos_46.to(
            device=device(type="cuda", index=0), dtype=torch.int64
        )
        relative_pos_46 = None
        getitem_27 = rel_embeddings_47[(slice(0, 512, None), slice(None, None, None))]
        rel_embeddings_47 = None
        rel_embeddings_48 = getitem_27.unsqueeze(0)
        getitem_27 = None
        linear_187 = torch._C._nn.linear(
            rel_embeddings_48,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_bias_ = (None)
        x_118 = linear_187.view((1, 512, 16, -1))
        linear_187 = None
        permute_141 = x_118.permute(0, 2, 1, 3)
        x_118 = None
        contiguous_141 = permute_141.contiguous()
        permute_141 = None
        view_329 = contiguous_141.view(-1, 512, 64)
        contiguous_141 = None
        pos_query_layer_23 = view_329.repeat(1, 1, 1)
        view_329 = None
        linear_188 = torch._C._nn.linear(
            rel_embeddings_48,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_bias_,
        )
        rel_embeddings_48 = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_bias_ = (None)
        x_119 = linear_188.view((1, 512, 16, -1))
        linear_188 = None
        permute_142 = x_119.permute(0, 2, 1, 3)
        x_119 = None
        contiguous_142 = permute_142.contiguous()
        permute_142 = None
        view_331 = contiguous_142.view(-1, 512, 64)
        contiguous_142 = None
        pos_key_layer_23 = view_331.repeat(1, 1, 1)
        view_331 = None
        tensor_72 = torch.tensor(64, dtype=torch.float32)
        mul_74 = tensor_72 * 3
        tensor_72 = None
        scale_70 = torch.sqrt(mul_74)
        mul_74 = None
        transpose_93 = pos_key_layer_23.transpose(-1, -2)
        pos_key_layer_23 = None
        c2p_att_46 = torch.bmm(query_layer_23, transpose_93)
        query_layer_23 = transpose_93 = None
        add_139 = relative_pos_47 + 256
        c2p_pos_23 = torch.clamp(add_139, 0, 511)
        add_139 = None
        squeeze_47 = c2p_pos_23.squeeze(0)
        c2p_pos_23 = None
        expand_46 = squeeze_47.expand([16, 11, 11])
        squeeze_47 = None
        c2p_att_47 = torch.gather(c2p_att_46, dim=-1, index=expand_46)
        c2p_att_46 = expand_46 = None
        to_96 = scale_70.to(dtype=torch.float32)
        scale_70 = None
        truediv_72 = c2p_att_47 / to_96
        c2p_att_47 = to_96 = None
        score_46 = 0 + truediv_72
        truediv_72 = None
        tensor_73 = torch.tensor(64, dtype=torch.float32)
        mul_75 = tensor_73 * 3
        tensor_73 = None
        scale_71 = torch.sqrt(mul_75)
        mul_75 = None
        neg_23 = -relative_pos_47
        relative_pos_47 = None
        add_141 = neg_23 + 256
        neg_23 = None
        p2c_pos_23 = torch.clamp(add_141, 0, 511)
        add_141 = None
        transpose_94 = pos_query_layer_23.transpose(-1, -2)
        pos_query_layer_23 = None
        p2c_att_46 = torch.bmm(key_layer_23, transpose_94)
        key_layer_23 = transpose_94 = None
        squeeze_48 = p2c_pos_23.squeeze(0)
        p2c_pos_23 = None
        expand_47 = squeeze_48.expand([16, 11, 11])
        squeeze_48 = None
        gather_47 = torch.gather(p2c_att_46, dim=-1, index=expand_47)
        p2c_att_46 = expand_47 = None
        p2c_att_47 = gather_47.transpose(-1, -2)
        gather_47 = None
        to_97 = scale_71.to(dtype=torch.float32)
        scale_71 = None
        truediv_73 = p2c_att_47 / to_97
        p2c_att_47 = to_97 = None
        score_46 += truediv_73
        score_47 = score_46
        score_46 = truediv_73 = None
        attention_scores_93 = attention_scores_92 + score_47
        attention_scores_92 = score_47 = None
        attention_scores_94 = attention_scores_93.view(-1, 16, 11, 11)
        attention_scores_93 = None
        attention_mask_24 = attention_mask.bool()
        attention_mask = None
        invert_23 = ~attention_mask_24
        attention_mask_24 = None
        attention_scores_95 = attention_scores_94.masked_fill(
            invert_23, -3.4028234663852886e38
        )
        attention_scores_94 = invert_23 = None
        attention_probs_46 = torch.nn.functional.softmax(attention_scores_95, dim=-1)
        attention_scores_95 = None
        attention_probs_47 = torch.nn.functional.dropout(
            attention_probs_46, 0.1, False, False
        )
        attention_probs_46 = None
        view_333 = attention_probs_47.view(-1, 11, 11)
        attention_probs_47 = None
        context_layer_69 = torch.bmm(view_333, value_layer_23)
        view_333 = value_layer_23 = None
        view_334 = context_layer_69.view(-1, 16, 11, 64)
        context_layer_69 = None
        permute_143 = view_334.permute(0, 2, 1, 3)
        view_334 = None
        context_layer_70 = permute_143.contiguous()
        permute_143 = None
        context_layer_71 = context_layer_70.view((1, 11, -1))
        context_layer_70 = None
        hidden_states_184 = torch._C._nn.linear(
            context_layer_71,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_71 = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_185 = torch.nn.functional.dropout(
            hidden_states_184, 0.1, False, False
        )
        hidden_states_184 = None
        add_143 = hidden_states_185 + hidden_states_183
        hidden_states_185 = hidden_states_183 = None
        hidden_states_186 = torch.nn.functional.layer_norm(
            add_143,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_143 = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_187 = torch._C._nn.linear(
            hidden_states_186,
            l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_188 = torch._C._nn.gelu(hidden_states_187)
        hidden_states_187 = None
        hidden_states_189 = torch._C._nn.linear(
            hidden_states_188,
            l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_188 = l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_190 = torch.nn.functional.dropout(
            hidden_states_189, 0.1, False, False
        )
        hidden_states_189 = None
        add_144 = hidden_states_190 + hidden_states_186
        hidden_states_190 = hidden_states_186 = None
        hidden_states_191 = torch.nn.functional.layer_norm(
            add_144,
            (1024,),
            l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_layer_norm_parameters_bias_,
            1e-07,
        )
        add_144 = l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_layer_norm_parameters_bias_ = (None)
        return (hidden_states_191,)
