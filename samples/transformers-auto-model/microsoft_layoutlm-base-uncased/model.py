import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_kwargs_input_ids_: torch.Tensor,
        L_kwargs_attention_mask_: torch.Tensor,
        L_kwargs_token_type_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_buffers_position_ids_: torch.Tensor,
        L_self_modules_embeddings_modules_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_x_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_y_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_h_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_w_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_pooler_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_kwargs_input_ids_ = L_kwargs_input_ids_
        l_kwargs_attention_mask_ = L_kwargs_attention_mask_
        l_kwargs_token_type_ids_ = L_kwargs_token_type_ids_
        l_self_modules_embeddings_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_buffers_position_ids_ = (
            L_self_modules_embeddings_buffers_position_ids_
        )
        l_self_modules_embeddings_modules_position_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_position_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_x_position_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_x_position_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_y_position_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_y_position_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_h_position_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_h_position_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_w_position_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_w_position_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_weight_ = (
            L_self_modules_embeddings_modules_LayerNorm_parameters_weight_
        )
        l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = (
            L_self_modules_embeddings_modules_LayerNorm_parameters_bias_
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_
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
        l_self_modules_pooler_modules_dense_parameters_weight_ = (
            L_self_modules_pooler_modules_dense_parameters_weight_
        )
        l_self_modules_pooler_modules_dense_parameters_bias_ = (
            L_self_modules_pooler_modules_dense_parameters_bias_
        )
        bbox = torch.zeros(
            (1, 11, 4), dtype=torch.int64, device=device(type="cuda", index=0)
        )
        unsqueeze = l_kwargs_attention_mask_.unsqueeze(1)
        l_kwargs_attention_mask_ = None
        extended_attention_mask = unsqueeze.unsqueeze(2)
        unsqueeze = None
        extended_attention_mask_1 = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = None
        sub = 1.0 - extended_attention_mask_1
        extended_attention_mask_1 = None
        extended_attention_mask_2 = sub * -3.4028234663852886e38
        sub = None
        position_ids = l_self_modules_embeddings_buffers_position_ids_[
            (slice(None, None, None), slice(None, 11, None))
        ]
        l_self_modules_embeddings_buffers_position_ids_ = None
        inputs_embeds = torch.nn.functional.embedding(
            l_kwargs_input_ids_,
            l_self_modules_embeddings_modules_word_embeddings_parameters_weight_,
            0,
            None,
            2.0,
            False,
            False,
        )
        l_kwargs_input_ids_ = (
            l_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        ) = None
        position_embeddings = torch.nn.functional.embedding(
            position_ids,
            l_self_modules_embeddings_modules_position_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        position_ids = (
            l_self_modules_embeddings_modules_position_embeddings_parameters_weight_
        ) = None
        getitem_1 = bbox[(slice(None, None, None), slice(None, None, None), 0)]
        left_position_embeddings = torch.nn.functional.embedding(
            getitem_1,
            l_self_modules_embeddings_modules_x_position_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        getitem_1 = None
        getitem_2 = bbox[(slice(None, None, None), slice(None, None, None), 1)]
        upper_position_embeddings = torch.nn.functional.embedding(
            getitem_2,
            l_self_modules_embeddings_modules_y_position_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        getitem_2 = None
        getitem_3 = bbox[(slice(None, None, None), slice(None, None, None), 2)]
        right_position_embeddings = torch.nn.functional.embedding(
            getitem_3,
            l_self_modules_embeddings_modules_x_position_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        getitem_3 = (
            l_self_modules_embeddings_modules_x_position_embeddings_parameters_weight_
        ) = None
        getitem_4 = bbox[(slice(None, None, None), slice(None, None, None), 3)]
        lower_position_embeddings = torch.nn.functional.embedding(
            getitem_4,
            l_self_modules_embeddings_modules_y_position_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        getitem_4 = (
            l_self_modules_embeddings_modules_y_position_embeddings_parameters_weight_
        ) = None
        getitem_5 = bbox[(slice(None, None, None), slice(None, None, None), 3)]
        getitem_6 = bbox[(slice(None, None, None), slice(None, None, None), 1)]
        sub_1 = getitem_5 - getitem_6
        getitem_5 = getitem_6 = None
        h_position_embeddings = torch.nn.functional.embedding(
            sub_1,
            l_self_modules_embeddings_modules_h_position_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        sub_1 = (
            l_self_modules_embeddings_modules_h_position_embeddings_parameters_weight_
        ) = None
        getitem_7 = bbox[(slice(None, None, None), slice(None, None, None), 2)]
        getitem_8 = bbox[(slice(None, None, None), slice(None, None, None), 0)]
        bbox = None
        sub_2 = getitem_7 - getitem_8
        getitem_7 = getitem_8 = None
        w_position_embeddings = torch.nn.functional.embedding(
            sub_2,
            l_self_modules_embeddings_modules_w_position_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        sub_2 = (
            l_self_modules_embeddings_modules_w_position_embeddings_parameters_weight_
        ) = None
        token_type_embeddings = torch.nn.functional.embedding(
            l_kwargs_token_type_ids_,
            l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        l_kwargs_token_type_ids_ = (
            l_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_
        ) = None
        add = inputs_embeds + position_embeddings
        inputs_embeds = position_embeddings = None
        add_1 = add + left_position_embeddings
        add = left_position_embeddings = None
        add_2 = add_1 + upper_position_embeddings
        add_1 = upper_position_embeddings = None
        add_3 = add_2 + right_position_embeddings
        add_2 = right_position_embeddings = None
        add_4 = add_3 + lower_position_embeddings
        add_3 = lower_position_embeddings = None
        add_5 = add_4 + h_position_embeddings
        add_4 = h_position_embeddings = None
        add_6 = add_5 + w_position_embeddings
        add_5 = w_position_embeddings = None
        embeddings = add_6 + token_type_embeddings
        add_6 = token_type_embeddings = None
        embeddings_1 = torch.nn.functional.layer_norm(
            embeddings,
            (768,),
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_,
            l_self_modules_embeddings_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        embeddings = (
            l_self_modules_embeddings_modules_layer_norm_parameters_weight_
        ) = l_self_modules_embeddings_modules_layer_norm_parameters_bias_ = None
        embeddings_2 = torch.nn.functional.dropout(embeddings_1, 0.1, False, False)
        embeddings_1 = None
        linear = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view = linear.view((1, 11, -1, 64))
        linear = None
        query_states = view.transpose(1, 2)
        view = None
        linear_1 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_1 = linear_1.view((1, 11, -1, 64))
        linear_1 = None
        key_states = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            embeddings_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_2 = linear_2.view((1, 11, -1, 64))
        linear_2 = None
        value_states = view_2.transpose(1, 2)
        view_2 = None
        transpose_3 = key_states.transpose(2, 3)
        key_states = None
        matmul = torch.matmul(query_states, transpose_3)
        query_states = transpose_3 = None
        attn_weights = matmul * 0.125
        matmul = None
        causal_mask = extended_attention_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        attn_weights_1 = attn_weights + causal_mask
        attn_weights = causal_mask = None
        softmax = torch.nn.functional.softmax(
            attn_weights_1, dim=-1, dtype=torch.float32
        )
        attn_weights_1 = None
        attn_weights_2 = softmax.to(torch.float32)
        softmax = None
        attn_weights_3 = torch.nn.functional.dropout(
            attn_weights_2, p=0.0, training=False
        )
        attn_weights_2 = None
        attn_output = torch.matmul(attn_weights_3, value_states)
        attn_weights_3 = value_states = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        reshape = attn_output_1.reshape(1, 11, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        hidden_states = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_2 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.1, False, False)
        hidden_states = None
        add_9 = hidden_states_1 + embeddings_2
        hidden_states_1 = embeddings_2 = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            add_9,
            (768,),
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_9 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_10 = hidden_states_6 + hidden_states_2
        hidden_states_6 = hidden_states_2 = None
        hidden_states_7 = torch.nn.functional.layer_norm(
            add_10,
            (768,),
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_10 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_3 = linear_6.view((1, 11, -1, 64))
        linear_6 = None
        query_states_1 = view_3.transpose(1, 2)
        view_3 = None
        linear_7 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_4 = linear_7.view((1, 11, -1, 64))
        linear_7 = None
        key_states_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_5 = linear_8.view((1, 11, -1, 64))
        linear_8 = None
        value_states_1 = view_5.transpose(1, 2)
        view_5 = None
        transpose_8 = key_states_1.transpose(2, 3)
        key_states_1 = None
        matmul_2 = torch.matmul(query_states_1, transpose_8)
        query_states_1 = transpose_8 = None
        attn_weights_4 = matmul_2 * 0.125
        matmul_2 = None
        causal_mask_1 = extended_attention_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        attn_weights_5 = attn_weights_4 + causal_mask_1
        attn_weights_4 = causal_mask_1 = None
        softmax_1 = torch.nn.functional.softmax(
            attn_weights_5, dim=-1, dtype=torch.float32
        )
        attn_weights_5 = None
        attn_weights_6 = softmax_1.to(torch.float32)
        softmax_1 = None
        attn_weights_7 = torch.nn.functional.dropout(
            attn_weights_6, p=0.0, training=False
        )
        attn_weights_6 = None
        attn_output_3 = torch.matmul(attn_weights_7, value_states_1)
        attn_weights_7 = value_states_1 = None
        transpose_9 = attn_output_3.transpose(1, 2)
        attn_output_3 = None
        attn_output_4 = transpose_9.contiguous()
        transpose_9 = None
        reshape_1 = attn_output_4.reshape(1, 11, -1)
        attn_output_4 = None
        attn_output_5 = reshape_1.contiguous()
        reshape_1 = None
        hidden_states_8 = torch._C._nn.linear(
            attn_output_5,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_5 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, 0.1, False, False
        )
        hidden_states_8 = None
        add_12 = hidden_states_9 + hidden_states_7
        hidden_states_9 = hidden_states_7 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            add_12,
            (768,),
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_12 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_13 = hidden_states_14 + hidden_states_10
        hidden_states_14 = hidden_states_10 = None
        hidden_states_15 = torch.nn.functional.layer_norm(
            add_13,
            (768,),
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_13 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_6 = linear_12.view((1, 11, -1, 64))
        linear_12 = None
        query_states_2 = view_6.transpose(1, 2)
        view_6 = None
        linear_13 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_7 = linear_13.view((1, 11, -1, 64))
        linear_13 = None
        key_states_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_8 = linear_14.view((1, 11, -1, 64))
        linear_14 = None
        value_states_2 = view_8.transpose(1, 2)
        view_8 = None
        transpose_13 = key_states_2.transpose(2, 3)
        key_states_2 = None
        matmul_4 = torch.matmul(query_states_2, transpose_13)
        query_states_2 = transpose_13 = None
        attn_weights_8 = matmul_4 * 0.125
        matmul_4 = None
        causal_mask_2 = extended_attention_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        attn_weights_9 = attn_weights_8 + causal_mask_2
        attn_weights_8 = causal_mask_2 = None
        softmax_2 = torch.nn.functional.softmax(
            attn_weights_9, dim=-1, dtype=torch.float32
        )
        attn_weights_9 = None
        attn_weights_10 = softmax_2.to(torch.float32)
        softmax_2 = None
        attn_weights_11 = torch.nn.functional.dropout(
            attn_weights_10, p=0.0, training=False
        )
        attn_weights_10 = None
        attn_output_6 = torch.matmul(attn_weights_11, value_states_2)
        attn_weights_11 = value_states_2 = None
        transpose_14 = attn_output_6.transpose(1, 2)
        attn_output_6 = None
        attn_output_7 = transpose_14.contiguous()
        transpose_14 = None
        reshape_2 = attn_output_7.reshape(1, 11, -1)
        attn_output_7 = None
        attn_output_8 = reshape_2.contiguous()
        reshape_2 = None
        hidden_states_16 = torch._C._nn.linear(
            attn_output_8,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_8 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_17 = torch.nn.functional.dropout(
            hidden_states_16, 0.1, False, False
        )
        hidden_states_16 = None
        add_15 = hidden_states_17 + hidden_states_15
        hidden_states_17 = hidden_states_15 = None
        hidden_states_18 = torch.nn.functional.layer_norm(
            add_15,
            (768,),
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_15 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_16 = hidden_states_22 + hidden_states_18
        hidden_states_22 = hidden_states_18 = None
        hidden_states_23 = torch.nn.functional.layer_norm(
            add_16,
            (768,),
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_16 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_9 = linear_18.view((1, 11, -1, 64))
        linear_18 = None
        query_states_3 = view_9.transpose(1, 2)
        view_9 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_10 = linear_19.view((1, 11, -1, 64))
        linear_19 = None
        key_states_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_20 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_11 = linear_20.view((1, 11, -1, 64))
        linear_20 = None
        value_states_3 = view_11.transpose(1, 2)
        view_11 = None
        transpose_18 = key_states_3.transpose(2, 3)
        key_states_3 = None
        matmul_6 = torch.matmul(query_states_3, transpose_18)
        query_states_3 = transpose_18 = None
        attn_weights_12 = matmul_6 * 0.125
        matmul_6 = None
        causal_mask_3 = extended_attention_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        attn_weights_13 = attn_weights_12 + causal_mask_3
        attn_weights_12 = causal_mask_3 = None
        softmax_3 = torch.nn.functional.softmax(
            attn_weights_13, dim=-1, dtype=torch.float32
        )
        attn_weights_13 = None
        attn_weights_14 = softmax_3.to(torch.float32)
        softmax_3 = None
        attn_weights_15 = torch.nn.functional.dropout(
            attn_weights_14, p=0.0, training=False
        )
        attn_weights_14 = None
        attn_output_9 = torch.matmul(attn_weights_15, value_states_3)
        attn_weights_15 = value_states_3 = None
        transpose_19 = attn_output_9.transpose(1, 2)
        attn_output_9 = None
        attn_output_10 = transpose_19.contiguous()
        transpose_19 = None
        reshape_3 = attn_output_10.reshape(1, 11, -1)
        attn_output_10 = None
        attn_output_11 = reshape_3.contiguous()
        reshape_3 = None
        hidden_states_24 = torch._C._nn.linear(
            attn_output_11,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_11 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, 0.1, False, False
        )
        hidden_states_24 = None
        add_18 = hidden_states_25 + hidden_states_23
        hidden_states_25 = hidden_states_23 = None
        hidden_states_26 = torch.nn.functional.layer_norm(
            add_18,
            (768,),
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_18 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_19 = hidden_states_30 + hidden_states_26
        hidden_states_30 = hidden_states_26 = None
        hidden_states_31 = torch.nn.functional.layer_norm(
            add_19,
            (768,),
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_19 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_12 = linear_24.view((1, 11, -1, 64))
        linear_24 = None
        query_states_4 = view_12.transpose(1, 2)
        view_12 = None
        linear_25 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_13 = linear_25.view((1, 11, -1, 64))
        linear_25 = None
        key_states_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_14 = linear_26.view((1, 11, -1, 64))
        linear_26 = None
        value_states_4 = view_14.transpose(1, 2)
        view_14 = None
        transpose_23 = key_states_4.transpose(2, 3)
        key_states_4 = None
        matmul_8 = torch.matmul(query_states_4, transpose_23)
        query_states_4 = transpose_23 = None
        attn_weights_16 = matmul_8 * 0.125
        matmul_8 = None
        causal_mask_4 = extended_attention_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        attn_weights_17 = attn_weights_16 + causal_mask_4
        attn_weights_16 = causal_mask_4 = None
        softmax_4 = torch.nn.functional.softmax(
            attn_weights_17, dim=-1, dtype=torch.float32
        )
        attn_weights_17 = None
        attn_weights_18 = softmax_4.to(torch.float32)
        softmax_4 = None
        attn_weights_19 = torch.nn.functional.dropout(
            attn_weights_18, p=0.0, training=False
        )
        attn_weights_18 = None
        attn_output_12 = torch.matmul(attn_weights_19, value_states_4)
        attn_weights_19 = value_states_4 = None
        transpose_24 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_24.contiguous()
        transpose_24 = None
        reshape_4 = attn_output_13.reshape(1, 11, -1)
        attn_output_13 = None
        attn_output_14 = reshape_4.contiguous()
        reshape_4 = None
        hidden_states_32 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_14 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_33 = torch.nn.functional.dropout(
            hidden_states_32, 0.1, False, False
        )
        hidden_states_32 = None
        add_21 = hidden_states_33 + hidden_states_31
        hidden_states_33 = hidden_states_31 = None
        hidden_states_34 = torch.nn.functional.layer_norm(
            add_21,
            (768,),
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_21 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_22 = hidden_states_38 + hidden_states_34
        hidden_states_38 = hidden_states_34 = None
        hidden_states_39 = torch.nn.functional.layer_norm(
            add_22,
            (768,),
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_22 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_30 = torch._C._nn.linear(
            hidden_states_39,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_15 = linear_30.view((1, 11, -1, 64))
        linear_30 = None
        query_states_5 = view_15.transpose(1, 2)
        view_15 = None
        linear_31 = torch._C._nn.linear(
            hidden_states_39,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_16 = linear_31.view((1, 11, -1, 64))
        linear_31 = None
        key_states_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_32 = torch._C._nn.linear(
            hidden_states_39,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_17 = linear_32.view((1, 11, -1, 64))
        linear_32 = None
        value_states_5 = view_17.transpose(1, 2)
        view_17 = None
        transpose_28 = key_states_5.transpose(2, 3)
        key_states_5 = None
        matmul_10 = torch.matmul(query_states_5, transpose_28)
        query_states_5 = transpose_28 = None
        attn_weights_20 = matmul_10 * 0.125
        matmul_10 = None
        causal_mask_5 = extended_attention_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        attn_weights_21 = attn_weights_20 + causal_mask_5
        attn_weights_20 = causal_mask_5 = None
        softmax_5 = torch.nn.functional.softmax(
            attn_weights_21, dim=-1, dtype=torch.float32
        )
        attn_weights_21 = None
        attn_weights_22 = softmax_5.to(torch.float32)
        softmax_5 = None
        attn_weights_23 = torch.nn.functional.dropout(
            attn_weights_22, p=0.0, training=False
        )
        attn_weights_22 = None
        attn_output_15 = torch.matmul(attn_weights_23, value_states_5)
        attn_weights_23 = value_states_5 = None
        transpose_29 = attn_output_15.transpose(1, 2)
        attn_output_15 = None
        attn_output_16 = transpose_29.contiguous()
        transpose_29 = None
        reshape_5 = attn_output_16.reshape(1, 11, -1)
        attn_output_16 = None
        attn_output_17 = reshape_5.contiguous()
        reshape_5 = None
        hidden_states_40 = torch._C._nn.linear(
            attn_output_17,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_17 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_41 = torch.nn.functional.dropout(
            hidden_states_40, 0.1, False, False
        )
        hidden_states_40 = None
        add_24 = hidden_states_41 + hidden_states_39
        hidden_states_41 = hidden_states_39 = None
        hidden_states_42 = torch.nn.functional.layer_norm(
            add_24,
            (768,),
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_24 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_25 = hidden_states_46 + hidden_states_42
        hidden_states_46 = hidden_states_42 = None
        hidden_states_47 = torch.nn.functional.layer_norm(
            add_25,
            (768,),
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_25 = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_36 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_18 = linear_36.view((1, 11, -1, 64))
        linear_36 = None
        query_states_6 = view_18.transpose(1, 2)
        view_18 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_19 = linear_37.view((1, 11, -1, 64))
        linear_37 = None
        key_states_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_38 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_20 = linear_38.view((1, 11, -1, 64))
        linear_38 = None
        value_states_6 = view_20.transpose(1, 2)
        view_20 = None
        transpose_33 = key_states_6.transpose(2, 3)
        key_states_6 = None
        matmul_12 = torch.matmul(query_states_6, transpose_33)
        query_states_6 = transpose_33 = None
        attn_weights_24 = matmul_12 * 0.125
        matmul_12 = None
        causal_mask_6 = extended_attention_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        attn_weights_25 = attn_weights_24 + causal_mask_6
        attn_weights_24 = causal_mask_6 = None
        softmax_6 = torch.nn.functional.softmax(
            attn_weights_25, dim=-1, dtype=torch.float32
        )
        attn_weights_25 = None
        attn_weights_26 = softmax_6.to(torch.float32)
        softmax_6 = None
        attn_weights_27 = torch.nn.functional.dropout(
            attn_weights_26, p=0.0, training=False
        )
        attn_weights_26 = None
        attn_output_18 = torch.matmul(attn_weights_27, value_states_6)
        attn_weights_27 = value_states_6 = None
        transpose_34 = attn_output_18.transpose(1, 2)
        attn_output_18 = None
        attn_output_19 = transpose_34.contiguous()
        transpose_34 = None
        reshape_6 = attn_output_19.reshape(1, 11, -1)
        attn_output_19 = None
        attn_output_20 = reshape_6.contiguous()
        reshape_6 = None
        hidden_states_48 = torch._C._nn.linear(
            attn_output_20,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_20 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_49 = torch.nn.functional.dropout(
            hidden_states_48, 0.1, False, False
        )
        hidden_states_48 = None
        add_27 = hidden_states_49 + hidden_states_47
        hidden_states_49 = hidden_states_47 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            add_27,
            (768,),
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_27 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_28 = hidden_states_54 + hidden_states_50
        hidden_states_54 = hidden_states_50 = None
        hidden_states_55 = torch.nn.functional.layer_norm(
            add_28,
            (768,),
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_28 = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_42 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_21 = linear_42.view((1, 11, -1, 64))
        linear_42 = None
        query_states_7 = view_21.transpose(1, 2)
        view_21 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_22 = linear_43.view((1, 11, -1, 64))
        linear_43 = None
        key_states_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_23 = linear_44.view((1, 11, -1, 64))
        linear_44 = None
        value_states_7 = view_23.transpose(1, 2)
        view_23 = None
        transpose_38 = key_states_7.transpose(2, 3)
        key_states_7 = None
        matmul_14 = torch.matmul(query_states_7, transpose_38)
        query_states_7 = transpose_38 = None
        attn_weights_28 = matmul_14 * 0.125
        matmul_14 = None
        causal_mask_7 = extended_attention_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        attn_weights_29 = attn_weights_28 + causal_mask_7
        attn_weights_28 = causal_mask_7 = None
        softmax_7 = torch.nn.functional.softmax(
            attn_weights_29, dim=-1, dtype=torch.float32
        )
        attn_weights_29 = None
        attn_weights_30 = softmax_7.to(torch.float32)
        softmax_7 = None
        attn_weights_31 = torch.nn.functional.dropout(
            attn_weights_30, p=0.0, training=False
        )
        attn_weights_30 = None
        attn_output_21 = torch.matmul(attn_weights_31, value_states_7)
        attn_weights_31 = value_states_7 = None
        transpose_39 = attn_output_21.transpose(1, 2)
        attn_output_21 = None
        attn_output_22 = transpose_39.contiguous()
        transpose_39 = None
        reshape_7 = attn_output_22.reshape(1, 11, -1)
        attn_output_22 = None
        attn_output_23 = reshape_7.contiguous()
        reshape_7 = None
        hidden_states_56 = torch._C._nn.linear(
            attn_output_23,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_23 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_57 = torch.nn.functional.dropout(
            hidden_states_56, 0.1, False, False
        )
        hidden_states_56 = None
        add_30 = hidden_states_57 + hidden_states_55
        hidden_states_57 = hidden_states_55 = None
        hidden_states_58 = torch.nn.functional.layer_norm(
            add_30,
            (768,),
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_30 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_31 = hidden_states_62 + hidden_states_58
        hidden_states_62 = hidden_states_58 = None
        hidden_states_63 = torch.nn.functional.layer_norm(
            add_31,
            (768,),
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_31 = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            hidden_states_63,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_24 = linear_48.view((1, 11, -1, 64))
        linear_48 = None
        query_states_8 = view_24.transpose(1, 2)
        view_24 = None
        linear_49 = torch._C._nn.linear(
            hidden_states_63,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_25 = linear_49.view((1, 11, -1, 64))
        linear_49 = None
        key_states_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_63,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_26 = linear_50.view((1, 11, -1, 64))
        linear_50 = None
        value_states_8 = view_26.transpose(1, 2)
        view_26 = None
        transpose_43 = key_states_8.transpose(2, 3)
        key_states_8 = None
        matmul_16 = torch.matmul(query_states_8, transpose_43)
        query_states_8 = transpose_43 = None
        attn_weights_32 = matmul_16 * 0.125
        matmul_16 = None
        causal_mask_8 = extended_attention_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        attn_weights_33 = attn_weights_32 + causal_mask_8
        attn_weights_32 = causal_mask_8 = None
        softmax_8 = torch.nn.functional.softmax(
            attn_weights_33, dim=-1, dtype=torch.float32
        )
        attn_weights_33 = None
        attn_weights_34 = softmax_8.to(torch.float32)
        softmax_8 = None
        attn_weights_35 = torch.nn.functional.dropout(
            attn_weights_34, p=0.0, training=False
        )
        attn_weights_34 = None
        attn_output_24 = torch.matmul(attn_weights_35, value_states_8)
        attn_weights_35 = value_states_8 = None
        transpose_44 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_44.contiguous()
        transpose_44 = None
        reshape_8 = attn_output_25.reshape(1, 11, -1)
        attn_output_25 = None
        attn_output_26 = reshape_8.contiguous()
        reshape_8 = None
        hidden_states_64 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_26 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_65 = torch.nn.functional.dropout(
            hidden_states_64, 0.1, False, False
        )
        hidden_states_64 = None
        add_33 = hidden_states_65 + hidden_states_63
        hidden_states_65 = hidden_states_63 = None
        hidden_states_66 = torch.nn.functional.layer_norm(
            add_33,
            (768,),
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_33 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_34 = hidden_states_70 + hidden_states_66
        hidden_states_70 = hidden_states_66 = None
        hidden_states_71 = torch.nn.functional.layer_norm(
            add_34,
            (768,),
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_34 = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_27 = linear_54.view((1, 11, -1, 64))
        linear_54 = None
        query_states_9 = view_27.transpose(1, 2)
        view_27 = None
        linear_55 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_28 = linear_55.view((1, 11, -1, 64))
        linear_55 = None
        key_states_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_56 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_29 = linear_56.view((1, 11, -1, 64))
        linear_56 = None
        value_states_9 = view_29.transpose(1, 2)
        view_29 = None
        transpose_48 = key_states_9.transpose(2, 3)
        key_states_9 = None
        matmul_18 = torch.matmul(query_states_9, transpose_48)
        query_states_9 = transpose_48 = None
        attn_weights_36 = matmul_18 * 0.125
        matmul_18 = None
        causal_mask_9 = extended_attention_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        attn_weights_37 = attn_weights_36 + causal_mask_9
        attn_weights_36 = causal_mask_9 = None
        softmax_9 = torch.nn.functional.softmax(
            attn_weights_37, dim=-1, dtype=torch.float32
        )
        attn_weights_37 = None
        attn_weights_38 = softmax_9.to(torch.float32)
        softmax_9 = None
        attn_weights_39 = torch.nn.functional.dropout(
            attn_weights_38, p=0.0, training=False
        )
        attn_weights_38 = None
        attn_output_27 = torch.matmul(attn_weights_39, value_states_9)
        attn_weights_39 = value_states_9 = None
        transpose_49 = attn_output_27.transpose(1, 2)
        attn_output_27 = None
        attn_output_28 = transpose_49.contiguous()
        transpose_49 = None
        reshape_9 = attn_output_28.reshape(1, 11, -1)
        attn_output_28 = None
        attn_output_29 = reshape_9.contiguous()
        reshape_9 = None
        hidden_states_72 = torch._C._nn.linear(
            attn_output_29,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_29 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_73 = torch.nn.functional.dropout(
            hidden_states_72, 0.1, False, False
        )
        hidden_states_72 = None
        add_36 = hidden_states_73 + hidden_states_71
        hidden_states_73 = hidden_states_71 = None
        hidden_states_74 = torch.nn.functional.layer_norm(
            add_36,
            (768,),
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_36 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_37 = hidden_states_78 + hidden_states_74
        hidden_states_78 = hidden_states_74 = None
        hidden_states_79 = torch.nn.functional.layer_norm(
            add_37,
            (768,),
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_37 = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_30 = linear_60.view((1, 11, -1, 64))
        linear_60 = None
        query_states_10 = view_30.transpose(1, 2)
        view_30 = None
        linear_61 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_31 = linear_61.view((1, 11, -1, 64))
        linear_61 = None
        key_states_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_62 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_32 = linear_62.view((1, 11, -1, 64))
        linear_62 = None
        value_states_10 = view_32.transpose(1, 2)
        view_32 = None
        transpose_53 = key_states_10.transpose(2, 3)
        key_states_10 = None
        matmul_20 = torch.matmul(query_states_10, transpose_53)
        query_states_10 = transpose_53 = None
        attn_weights_40 = matmul_20 * 0.125
        matmul_20 = None
        causal_mask_10 = extended_attention_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        attn_weights_41 = attn_weights_40 + causal_mask_10
        attn_weights_40 = causal_mask_10 = None
        softmax_10 = torch.nn.functional.softmax(
            attn_weights_41, dim=-1, dtype=torch.float32
        )
        attn_weights_41 = None
        attn_weights_42 = softmax_10.to(torch.float32)
        softmax_10 = None
        attn_weights_43 = torch.nn.functional.dropout(
            attn_weights_42, p=0.0, training=False
        )
        attn_weights_42 = None
        attn_output_30 = torch.matmul(attn_weights_43, value_states_10)
        attn_weights_43 = value_states_10 = None
        transpose_54 = attn_output_30.transpose(1, 2)
        attn_output_30 = None
        attn_output_31 = transpose_54.contiguous()
        transpose_54 = None
        reshape_10 = attn_output_31.reshape(1, 11, -1)
        attn_output_31 = None
        attn_output_32 = reshape_10.contiguous()
        reshape_10 = None
        hidden_states_80 = torch._C._nn.linear(
            attn_output_32,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_32 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_81 = torch.nn.functional.dropout(
            hidden_states_80, 0.1, False, False
        )
        hidden_states_80 = None
        add_39 = hidden_states_81 + hidden_states_79
        hidden_states_81 = hidden_states_79 = None
        hidden_states_82 = torch.nn.functional.layer_norm(
            add_39,
            (768,),
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_39 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_40 = hidden_states_86 + hidden_states_82
        hidden_states_86 = hidden_states_82 = None
        hidden_states_87 = torch.nn.functional.layer_norm(
            add_40,
            (768,),
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_40 = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_bias_ = (None)
        linear_66 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_33 = linear_66.view((1, 11, -1, 64))
        linear_66 = None
        query_states_11 = view_33.transpose(1, 2)
        view_33 = None
        linear_67 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_34 = linear_67.view((1, 11, -1, 64))
        linear_67 = None
        key_states_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_35 = linear_68.view((1, 11, -1, 64))
        linear_68 = None
        value_states_11 = view_35.transpose(1, 2)
        view_35 = None
        transpose_58 = key_states_11.transpose(2, 3)
        key_states_11 = None
        matmul_22 = torch.matmul(query_states_11, transpose_58)
        query_states_11 = transpose_58 = None
        attn_weights_44 = matmul_22 * 0.125
        matmul_22 = None
        causal_mask_11 = extended_attention_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 11, None),
            )
        ]
        extended_attention_mask_2 = None
        attn_weights_45 = attn_weights_44 + causal_mask_11
        attn_weights_44 = causal_mask_11 = None
        softmax_11 = torch.nn.functional.softmax(
            attn_weights_45, dim=-1, dtype=torch.float32
        )
        attn_weights_45 = None
        attn_weights_46 = softmax_11.to(torch.float32)
        softmax_11 = None
        attn_weights_47 = torch.nn.functional.dropout(
            attn_weights_46, p=0.0, training=False
        )
        attn_weights_46 = None
        attn_output_33 = torch.matmul(attn_weights_47, value_states_11)
        attn_weights_47 = value_states_11 = None
        transpose_59 = attn_output_33.transpose(1, 2)
        attn_output_33 = None
        attn_output_34 = transpose_59.contiguous()
        transpose_59 = None
        reshape_11 = attn_output_34.reshape(1, 11, -1)
        attn_output_34 = None
        attn_output_35 = reshape_11.contiguous()
        reshape_11 = None
        hidden_states_88 = torch._C._nn.linear(
            attn_output_35,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_35 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_89 = torch.nn.functional.dropout(
            hidden_states_88, 0.1, False, False
        )
        hidden_states_88 = None
        add_42 = hidden_states_89 + hidden_states_87
        hidden_states_89 = hidden_states_87 = None
        hidden_states_90 = torch.nn.functional.layer_norm(
            add_42,
            (768,),
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_42 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (None)
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
        add_43 = hidden_states_94 + hidden_states_90
        hidden_states_94 = hidden_states_90 = None
        hidden_states_95 = torch.nn.functional.layer_norm(
            add_43,
            (768,),
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_bias_,
            1e-12,
        )
        add_43 = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_bias_ = (None)
        first_token_tensor = hidden_states_95[(slice(None, None, None), 0)]
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
        return (hidden_states_95, pooled_output_1)
