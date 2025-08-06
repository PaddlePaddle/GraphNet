import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_kwargs_input_ids_: torch.Tensor,
        L_kwargs_attention_mask_: torch.Tensor,
        L_self_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_32_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_emb_layer_norm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_emb_layer_norm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_kwargs_input_ids_ = L_kwargs_input_ids_
        l_kwargs_attention_mask_ = L_kwargs_attention_mask_
        l_self_modules_embeddings_modules_word_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_word_embeddings_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_LayerNorm_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_LayerNorm_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_32_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_32_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_32_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_32_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_emb_layer_norm_after_parameters_weight_ = (
            L_self_modules_encoder_modules_emb_layer_norm_after_parameters_weight_
        )
        l_self_modules_encoder_modules_emb_layer_norm_after_parameters_bias_ = (
            L_self_modules_encoder_modules_emb_layer_norm_after_parameters_bias_
        )
        l_self_modules_pooler_modules_dense_parameters_weight_ = (
            L_self_modules_pooler_modules_dense_parameters_weight_
        )
        l_self_modules_pooler_modules_dense_parameters_bias_ = (
            L_self_modules_pooler_modules_dense_parameters_bias_
        )
        extended_attention_mask = l_kwargs_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        extended_attention_mask_1 = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = None
        sub = 1.0 - extended_attention_mask_1
        extended_attention_mask_1 = None
        extended_attention_mask_2 = sub * -3.4028234663852886e38
        sub = None
        ne = l_kwargs_input_ids_.ne(1)
        mask = ne.int()
        ne = None
        cumsum = torch.cumsum(mask, dim=1)
        type_as = cumsum.type_as(mask)
        cumsum = None
        incremental_indices = type_as * mask
        type_as = mask = None
        long = incremental_indices.long()
        incremental_indices = None
        position_ids = long + 1
        long = position_ids = None
        inputs_embeds = torch.nn.functional.embedding(
            l_kwargs_input_ids_,
            l_self_modules_embeddings_modules_word_embeddings_parameters_weight_,
            1,
            None,
            2.0,
            False,
            False,
        )
        l_self_modules_embeddings_modules_word_embeddings_parameters_weight_ = None
        eq = l_kwargs_input_ids_.__eq__(32)
        unsqueeze = eq.unsqueeze(-1)
        eq = None
        embeddings = inputs_embeds.masked_fill(unsqueeze, 0.0)
        inputs_embeds = unsqueeze = None
        src_lengths = l_kwargs_attention_mask_.sum(-1)
        eq_1 = l_kwargs_input_ids_.__eq__(32)
        l_kwargs_input_ids_ = None
        sum_2 = eq_1.sum(-1)
        eq_1 = None
        float_1 = sum_2.float()
        sum_2 = None
        mask_ratio_observed = float_1 / src_lengths
        float_1 = src_lengths = None
        mul_2 = embeddings * 0.88
        embeddings = None
        sub_1 = 1 - mask_ratio_observed
        mask_ratio_observed = None
        getitem_1 = sub_1[(slice(None, None, None), None, None)]
        sub_1 = None
        truediv_1 = mul_2 / getitem_1
        mul_2 = getitem_1 = None
        embeddings_1 = truediv_1.to(torch.float32)
        truediv_1 = None
        unsqueeze_1 = l_kwargs_attention_mask_.unsqueeze(-1)
        l_kwargs_attention_mask_ = None
        mul_3 = embeddings_1 * unsqueeze_1
        embeddings_1 = unsqueeze_1 = None
        embeddings_2 = mul_3.to(torch.float32)
        mul_3 = None
        hidden_states_ln = torch.nn.functional.layer_norm(
            embeddings_2,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear = torch._C._nn.linear(
            hidden_states_ln,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view = linear.view((1, -1, 20, 64))
        linear = None
        query_layer = view.transpose(1, 2)
        view = None
        linear_1 = torch._C._nn.linear(
            hidden_states_ln,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_1 = linear_1.view((1, -1, 20, 64))
        linear_1 = None
        key_layer = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            hidden_states_ln,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_2 = linear_2.view((1, -1, 20, 64))
        linear_2 = None
        value_layer = view_2.transpose(1, 2)
        view_2 = None
        query_layer_1 = query_layer * 0.125
        query_layer = None
        arange = torch.arange(13, device=device(type="cuda", index=0))
        t = arange.type_as(
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange = None
        freqs = torch.outer(
            t,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat = torch.cat((freqs, freqs), dim=-1)
        freqs = None
        emb = cat.to(device(type="cuda", index=0))
        cat = None
        cos = emb.cos()
        getitem_2 = cos[(None, None, slice(None, None, None), slice(None, None, None))]
        cos = None
        sin = emb.sin()
        emb = None
        getitem_3 = sin[(None, None, slice(None, None, None), slice(None, None, None))]
        sin = None
        cos_1 = getitem_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_1 = getitem_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_5 = query_layer_1 * cos_1
        cos_1 = None
        chunk = query_layer_1.chunk(2, dim=-1)
        query_layer_1 = None
        x1 = chunk[0]
        x2 = chunk[1]
        chunk = None
        neg = -x2
        x2 = None
        cat_1 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_6 = cat_1 * sin_1
        cat_1 = sin_1 = None
        add_1 = mul_5 + mul_6
        mul_5 = mul_6 = None
        query_layer_2 = add_1.to(dtype=torch.float32)
        add_1 = None
        cos_2 = getitem_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_2 = getitem_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_7 = key_layer * cos_2
        cos_2 = None
        chunk_1 = key_layer.chunk(2, dim=-1)
        key_layer = None
        x1_1 = chunk_1[0]
        x2_1 = chunk_1[1]
        chunk_1 = None
        neg_1 = -x2_1
        x2_1 = None
        cat_2 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_8 = cat_2 * sin_2
        cat_2 = sin_2 = None
        add_2 = mul_7 + mul_8
        mul_7 = mul_8 = None
        key_layer_1 = add_2.to(dtype=torch.float32)
        add_2 = None
        transpose_3 = key_layer_1.transpose(-1, -2)
        key_layer_1 = None
        attention_scores = torch.matmul(query_layer_2, transpose_3)
        query_layer_2 = transpose_3 = None
        attention_scores_1 = attention_scores + extended_attention_mask_2
        attention_scores = None
        attention_probs = torch.nn.functional.softmax(attention_scores_1, dim=-1)
        attention_scores_1 = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.0, False, False
        )
        attention_probs = None
        to_6 = attention_probs_1.to(torch.float32)
        attention_probs_1 = None
        context_layer = torch.matmul(to_6, value_layer)
        to_6 = value_layer = None
        permute = context_layer.permute(0, 2, 1, 3)
        context_layer = None
        context_layer_1 = permute.contiguous()
        permute = None
        context_layer_2 = context_layer_1.view((1, 13, 1280))
        context_layer_1 = None
        hidden_states = torch._C._nn.linear(
            context_layer_2,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.0, False, False)
        hidden_states = None
        hidden_states_2 = hidden_states_1 + embeddings_2
        hidden_states_1 = embeddings_2 = None
        attention_output_ln = torch.nn.functional.layer_norm(
            hidden_states_2,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_0_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_3 = torch._C._nn.linear(
            attention_output_ln,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln = l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_9 = hidden_states_3 * 0.5
        truediv_2 = hidden_states_3 / 1.4142135623730951
        hidden_states_3 = None
        erf = torch.erf(truediv_2)
        truediv_2 = None
        add_5 = 1.0 + erf
        erf = None
        hidden_states_4 = mul_9 * add_5
        mul_9 = add_5 = None
        hidden_states_5 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_4 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, 0.0, False, False
        )
        hidden_states_5 = None
        hidden_states_7 = hidden_states_6 + hidden_states_2
        hidden_states_6 = hidden_states_2 = None
        hidden_states_ln_1 = torch.nn.functional.layer_norm(
            hidden_states_7,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            hidden_states_ln_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_4 = linear_6.view((1, -1, 20, 64))
        linear_6 = None
        query_layer_3 = view_4.transpose(1, 2)
        view_4 = None
        linear_7 = torch._C._nn.linear(
            hidden_states_ln_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_5 = linear_7.view((1, -1, 20, 64))
        linear_7 = None
        key_layer_2 = view_5.transpose(1, 2)
        view_5 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_ln_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_1 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_6 = linear_8.view((1, -1, 20, 64))
        linear_8 = None
        value_layer_1 = view_6.transpose(1, 2)
        view_6 = None
        query_layer_4 = query_layer_3 * 0.125
        query_layer_3 = None
        arange_1 = torch.arange(13, device=device(type="cuda", index=0))
        t_1 = arange_1.type_as(
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_1 = None
        freqs_1 = torch.outer(
            t_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_1 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_3 = torch.cat((freqs_1, freqs_1), dim=-1)
        freqs_1 = None
        emb_1 = cat_3.to(device(type="cuda", index=0))
        cat_3 = None
        cos_3 = emb_1.cos()
        getitem_12 = cos_3[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_3 = None
        sin_3 = emb_1.sin()
        emb_1 = None
        getitem_13 = sin_3[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_3 = None
        cos_4 = getitem_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_4 = getitem_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_12 = query_layer_4 * cos_4
        cos_4 = None
        chunk_2 = query_layer_4.chunk(2, dim=-1)
        query_layer_4 = None
        x1_2 = chunk_2[0]
        x2_2 = chunk_2[1]
        chunk_2 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_4 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_13 = cat_4 * sin_4
        cat_4 = sin_4 = None
        add_7 = mul_12 + mul_13
        mul_12 = mul_13 = None
        query_layer_5 = add_7.to(dtype=torch.float32)
        add_7 = None
        cos_5 = getitem_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_5 = getitem_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_14 = key_layer_2 * cos_5
        cos_5 = None
        chunk_3 = key_layer_2.chunk(2, dim=-1)
        key_layer_2 = None
        x1_3 = chunk_3[0]
        x2_3 = chunk_3[1]
        chunk_3 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_5 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_15 = cat_5 * sin_5
        cat_5 = sin_5 = None
        add_8 = mul_14 + mul_15
        mul_14 = mul_15 = None
        key_layer_3 = add_8.to(dtype=torch.float32)
        add_8 = None
        transpose_7 = key_layer_3.transpose(-1, -2)
        key_layer_3 = None
        attention_scores_2 = torch.matmul(query_layer_5, transpose_7)
        query_layer_5 = transpose_7 = None
        attention_scores_3 = attention_scores_2 + extended_attention_mask_2
        attention_scores_2 = None
        attention_probs_2 = torch.nn.functional.softmax(attention_scores_3, dim=-1)
        attention_scores_3 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.0, False, False
        )
        attention_probs_2 = None
        to_10 = attention_probs_3.to(torch.float32)
        attention_probs_3 = None
        context_layer_3 = torch.matmul(to_10, value_layer_1)
        to_10 = value_layer_1 = None
        permute_1 = context_layer_3.permute(0, 2, 1, 3)
        context_layer_3 = None
        context_layer_4 = permute_1.contiguous()
        permute_1 = None
        context_layer_5 = context_layer_4.view((1, 13, 1280))
        context_layer_4 = None
        hidden_states_8 = torch._C._nn.linear(
            context_layer_5,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_5 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, 0.0, False, False
        )
        hidden_states_8 = None
        hidden_states_10 = hidden_states_9 + hidden_states_7
        hidden_states_9 = hidden_states_7 = None
        attention_output_ln_1 = torch.nn.functional.layer_norm(
            hidden_states_10,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_1_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_11 = torch._C._nn.linear(
            attention_output_ln_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_1 = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_16 = hidden_states_11 * 0.5
        truediv_3 = hidden_states_11 / 1.4142135623730951
        hidden_states_11 = None
        erf_1 = torch.erf(truediv_3)
        truediv_3 = None
        add_11 = 1.0 + erf_1
        erf_1 = None
        hidden_states_12 = mul_16 * add_11
        mul_16 = add_11 = None
        hidden_states_13 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_12 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_14 = torch.nn.functional.dropout(
            hidden_states_13, 0.0, False, False
        )
        hidden_states_13 = None
        hidden_states_15 = hidden_states_14 + hidden_states_10
        hidden_states_14 = hidden_states_10 = None
        hidden_states_ln_2 = torch.nn.functional.layer_norm(
            hidden_states_15,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            hidden_states_ln_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_8 = linear_12.view((1, -1, 20, 64))
        linear_12 = None
        query_layer_6 = view_8.transpose(1, 2)
        view_8 = None
        linear_13 = torch._C._nn.linear(
            hidden_states_ln_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_9 = linear_13.view((1, -1, 20, 64))
        linear_13 = None
        key_layer_4 = view_9.transpose(1, 2)
        view_9 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_ln_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_2 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_10 = linear_14.view((1, -1, 20, 64))
        linear_14 = None
        value_layer_2 = view_10.transpose(1, 2)
        view_10 = None
        query_layer_7 = query_layer_6 * 0.125
        query_layer_6 = None
        arange_2 = torch.arange(13, device=device(type="cuda", index=0))
        t_2 = arange_2.type_as(
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_2 = None
        freqs_2 = torch.outer(
            t_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_2 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_6 = torch.cat((freqs_2, freqs_2), dim=-1)
        freqs_2 = None
        emb_2 = cat_6.to(device(type="cuda", index=0))
        cat_6 = None
        cos_6 = emb_2.cos()
        getitem_22 = cos_6[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_6 = None
        sin_6 = emb_2.sin()
        emb_2 = None
        getitem_23 = sin_6[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_6 = None
        cos_7 = getitem_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_7 = getitem_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_19 = query_layer_7 * cos_7
        cos_7 = None
        chunk_4 = query_layer_7.chunk(2, dim=-1)
        query_layer_7 = None
        x1_4 = chunk_4[0]
        x2_4 = chunk_4[1]
        chunk_4 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_7 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_20 = cat_7 * sin_7
        cat_7 = sin_7 = None
        add_13 = mul_19 + mul_20
        mul_19 = mul_20 = None
        query_layer_8 = add_13.to(dtype=torch.float32)
        add_13 = None
        cos_8 = getitem_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_8 = getitem_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_21 = key_layer_4 * cos_8
        cos_8 = None
        chunk_5 = key_layer_4.chunk(2, dim=-1)
        key_layer_4 = None
        x1_5 = chunk_5[0]
        x2_5 = chunk_5[1]
        chunk_5 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_8 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_22 = cat_8 * sin_8
        cat_8 = sin_8 = None
        add_14 = mul_21 + mul_22
        mul_21 = mul_22 = None
        key_layer_5 = add_14.to(dtype=torch.float32)
        add_14 = None
        transpose_11 = key_layer_5.transpose(-1, -2)
        key_layer_5 = None
        attention_scores_4 = torch.matmul(query_layer_8, transpose_11)
        query_layer_8 = transpose_11 = None
        attention_scores_5 = attention_scores_4 + extended_attention_mask_2
        attention_scores_4 = None
        attention_probs_4 = torch.nn.functional.softmax(attention_scores_5, dim=-1)
        attention_scores_5 = None
        attention_probs_5 = torch.nn.functional.dropout(
            attention_probs_4, 0.0, False, False
        )
        attention_probs_4 = None
        to_14 = attention_probs_5.to(torch.float32)
        attention_probs_5 = None
        context_layer_6 = torch.matmul(to_14, value_layer_2)
        to_14 = value_layer_2 = None
        permute_2 = context_layer_6.permute(0, 2, 1, 3)
        context_layer_6 = None
        context_layer_7 = permute_2.contiguous()
        permute_2 = None
        context_layer_8 = context_layer_7.view((1, 13, 1280))
        context_layer_7 = None
        hidden_states_16 = torch._C._nn.linear(
            context_layer_8,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_8 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_17 = torch.nn.functional.dropout(
            hidden_states_16, 0.0, False, False
        )
        hidden_states_16 = None
        hidden_states_18 = hidden_states_17 + hidden_states_15
        hidden_states_17 = hidden_states_15 = None
        attention_output_ln_2 = torch.nn.functional.layer_norm(
            hidden_states_18,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_2_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_19 = torch._C._nn.linear(
            attention_output_ln_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_2 = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_23 = hidden_states_19 * 0.5
        truediv_4 = hidden_states_19 / 1.4142135623730951
        hidden_states_19 = None
        erf_2 = torch.erf(truediv_4)
        truediv_4 = None
        add_17 = 1.0 + erf_2
        erf_2 = None
        hidden_states_20 = mul_23 * add_17
        mul_23 = add_17 = None
        hidden_states_21 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, 0.0, False, False
        )
        hidden_states_21 = None
        hidden_states_23 = hidden_states_22 + hidden_states_18
        hidden_states_22 = hidden_states_18 = None
        hidden_states_ln_3 = torch.nn.functional.layer_norm(
            hidden_states_23,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            hidden_states_ln_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_12 = linear_18.view((1, -1, 20, 64))
        linear_18 = None
        query_layer_9 = view_12.transpose(1, 2)
        view_12 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_ln_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_13 = linear_19.view((1, -1, 20, 64))
        linear_19 = None
        key_layer_6 = view_13.transpose(1, 2)
        view_13 = None
        linear_20 = torch._C._nn.linear(
            hidden_states_ln_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_3 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_14 = linear_20.view((1, -1, 20, 64))
        linear_20 = None
        value_layer_3 = view_14.transpose(1, 2)
        view_14 = None
        query_layer_10 = query_layer_9 * 0.125
        query_layer_9 = None
        arange_3 = torch.arange(13, device=device(type="cuda", index=0))
        t_3 = arange_3.type_as(
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_3 = None
        freqs_3 = torch.outer(
            t_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_3 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_9 = torch.cat((freqs_3, freqs_3), dim=-1)
        freqs_3 = None
        emb_3 = cat_9.to(device(type="cuda", index=0))
        cat_9 = None
        cos_9 = emb_3.cos()
        getitem_32 = cos_9[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_9 = None
        sin_9 = emb_3.sin()
        emb_3 = None
        getitem_33 = sin_9[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_9 = None
        cos_10 = getitem_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_10 = getitem_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_26 = query_layer_10 * cos_10
        cos_10 = None
        chunk_6 = query_layer_10.chunk(2, dim=-1)
        query_layer_10 = None
        x1_6 = chunk_6[0]
        x2_6 = chunk_6[1]
        chunk_6 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_10 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_27 = cat_10 * sin_10
        cat_10 = sin_10 = None
        add_19 = mul_26 + mul_27
        mul_26 = mul_27 = None
        query_layer_11 = add_19.to(dtype=torch.float32)
        add_19 = None
        cos_11 = getitem_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_11 = getitem_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_28 = key_layer_6 * cos_11
        cos_11 = None
        chunk_7 = key_layer_6.chunk(2, dim=-1)
        key_layer_6 = None
        x1_7 = chunk_7[0]
        x2_7 = chunk_7[1]
        chunk_7 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_11 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_29 = cat_11 * sin_11
        cat_11 = sin_11 = None
        add_20 = mul_28 + mul_29
        mul_28 = mul_29 = None
        key_layer_7 = add_20.to(dtype=torch.float32)
        add_20 = None
        transpose_15 = key_layer_7.transpose(-1, -2)
        key_layer_7 = None
        attention_scores_6 = torch.matmul(query_layer_11, transpose_15)
        query_layer_11 = transpose_15 = None
        attention_scores_7 = attention_scores_6 + extended_attention_mask_2
        attention_scores_6 = None
        attention_probs_6 = torch.nn.functional.softmax(attention_scores_7, dim=-1)
        attention_scores_7 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.0, False, False
        )
        attention_probs_6 = None
        to_18 = attention_probs_7.to(torch.float32)
        attention_probs_7 = None
        context_layer_9 = torch.matmul(to_18, value_layer_3)
        to_18 = value_layer_3 = None
        permute_3 = context_layer_9.permute(0, 2, 1, 3)
        context_layer_9 = None
        context_layer_10 = permute_3.contiguous()
        permute_3 = None
        context_layer_11 = context_layer_10.view((1, 13, 1280))
        context_layer_10 = None
        hidden_states_24 = torch._C._nn.linear(
            context_layer_11,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, 0.0, False, False
        )
        hidden_states_24 = None
        hidden_states_26 = hidden_states_25 + hidden_states_23
        hidden_states_25 = hidden_states_23 = None
        attention_output_ln_3 = torch.nn.functional.layer_norm(
            hidden_states_26,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_3_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_27 = torch._C._nn.linear(
            attention_output_ln_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_3 = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_30 = hidden_states_27 * 0.5
        truediv_5 = hidden_states_27 / 1.4142135623730951
        hidden_states_27 = None
        erf_3 = torch.erf(truediv_5)
        truediv_5 = None
        add_23 = 1.0 + erf_3
        erf_3 = None
        hidden_states_28 = mul_30 * add_23
        mul_30 = add_23 = None
        hidden_states_29 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_28 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_30 = torch.nn.functional.dropout(
            hidden_states_29, 0.0, False, False
        )
        hidden_states_29 = None
        hidden_states_31 = hidden_states_30 + hidden_states_26
        hidden_states_30 = hidden_states_26 = None
        hidden_states_ln_4 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            hidden_states_ln_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_16 = linear_24.view((1, -1, 20, 64))
        linear_24 = None
        query_layer_12 = view_16.transpose(1, 2)
        view_16 = None
        linear_25 = torch._C._nn.linear(
            hidden_states_ln_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_17 = linear_25.view((1, -1, 20, 64))
        linear_25 = None
        key_layer_8 = view_17.transpose(1, 2)
        view_17 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_ln_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_4 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_18 = linear_26.view((1, -1, 20, 64))
        linear_26 = None
        value_layer_4 = view_18.transpose(1, 2)
        view_18 = None
        query_layer_13 = query_layer_12 * 0.125
        query_layer_12 = None
        arange_4 = torch.arange(13, device=device(type="cuda", index=0))
        t_4 = arange_4.type_as(
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_4 = None
        freqs_4 = torch.outer(
            t_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_4 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_12 = torch.cat((freqs_4, freqs_4), dim=-1)
        freqs_4 = None
        emb_4 = cat_12.to(device(type="cuda", index=0))
        cat_12 = None
        cos_12 = emb_4.cos()
        getitem_42 = cos_12[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_12 = None
        sin_12 = emb_4.sin()
        emb_4 = None
        getitem_43 = sin_12[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_12 = None
        cos_13 = getitem_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_13 = getitem_43[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_33 = query_layer_13 * cos_13
        cos_13 = None
        chunk_8 = query_layer_13.chunk(2, dim=-1)
        query_layer_13 = None
        x1_8 = chunk_8[0]
        x2_8 = chunk_8[1]
        chunk_8 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_13 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_34 = cat_13 * sin_13
        cat_13 = sin_13 = None
        add_25 = mul_33 + mul_34
        mul_33 = mul_34 = None
        query_layer_14 = add_25.to(dtype=torch.float32)
        add_25 = None
        cos_14 = getitem_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_14 = getitem_43[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_35 = key_layer_8 * cos_14
        cos_14 = None
        chunk_9 = key_layer_8.chunk(2, dim=-1)
        key_layer_8 = None
        x1_9 = chunk_9[0]
        x2_9 = chunk_9[1]
        chunk_9 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_14 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_36 = cat_14 * sin_14
        cat_14 = sin_14 = None
        add_26 = mul_35 + mul_36
        mul_35 = mul_36 = None
        key_layer_9 = add_26.to(dtype=torch.float32)
        add_26 = None
        transpose_19 = key_layer_9.transpose(-1, -2)
        key_layer_9 = None
        attention_scores_8 = torch.matmul(query_layer_14, transpose_19)
        query_layer_14 = transpose_19 = None
        attention_scores_9 = attention_scores_8 + extended_attention_mask_2
        attention_scores_8 = None
        attention_probs_8 = torch.nn.functional.softmax(attention_scores_9, dim=-1)
        attention_scores_9 = None
        attention_probs_9 = torch.nn.functional.dropout(
            attention_probs_8, 0.0, False, False
        )
        attention_probs_8 = None
        to_22 = attention_probs_9.to(torch.float32)
        attention_probs_9 = None
        context_layer_12 = torch.matmul(to_22, value_layer_4)
        to_22 = value_layer_4 = None
        permute_4 = context_layer_12.permute(0, 2, 1, 3)
        context_layer_12 = None
        context_layer_13 = permute_4.contiguous()
        permute_4 = None
        context_layer_14 = context_layer_13.view((1, 13, 1280))
        context_layer_13 = None
        hidden_states_32 = torch._C._nn.linear(
            context_layer_14,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_14 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_33 = torch.nn.functional.dropout(
            hidden_states_32, 0.0, False, False
        )
        hidden_states_32 = None
        hidden_states_34 = hidden_states_33 + hidden_states_31
        hidden_states_33 = hidden_states_31 = None
        attention_output_ln_4 = torch.nn.functional.layer_norm(
            hidden_states_34,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_4_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_35 = torch._C._nn.linear(
            attention_output_ln_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_4 = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_37 = hidden_states_35 * 0.5
        truediv_6 = hidden_states_35 / 1.4142135623730951
        hidden_states_35 = None
        erf_4 = torch.erf(truediv_6)
        truediv_6 = None
        add_29 = 1.0 + erf_4
        erf_4 = None
        hidden_states_36 = mul_37 * add_29
        mul_37 = add_29 = None
        hidden_states_37 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_36 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_38 = torch.nn.functional.dropout(
            hidden_states_37, 0.0, False, False
        )
        hidden_states_37 = None
        hidden_states_39 = hidden_states_38 + hidden_states_34
        hidden_states_38 = hidden_states_34 = None
        hidden_states_ln_5 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_30 = torch._C._nn.linear(
            hidden_states_ln_5,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_20 = linear_30.view((1, -1, 20, 64))
        linear_30 = None
        query_layer_15 = view_20.transpose(1, 2)
        view_20 = None
        linear_31 = torch._C._nn.linear(
            hidden_states_ln_5,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_21 = linear_31.view((1, -1, 20, 64))
        linear_31 = None
        key_layer_10 = view_21.transpose(1, 2)
        view_21 = None
        linear_32 = torch._C._nn.linear(
            hidden_states_ln_5,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_5 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_22 = linear_32.view((1, -1, 20, 64))
        linear_32 = None
        value_layer_5 = view_22.transpose(1, 2)
        view_22 = None
        query_layer_16 = query_layer_15 * 0.125
        query_layer_15 = None
        arange_5 = torch.arange(13, device=device(type="cuda", index=0))
        t_5 = arange_5.type_as(
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_5 = None
        freqs_5 = torch.outer(
            t_5,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_5 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_15 = torch.cat((freqs_5, freqs_5), dim=-1)
        freqs_5 = None
        emb_5 = cat_15.to(device(type="cuda", index=0))
        cat_15 = None
        cos_15 = emb_5.cos()
        getitem_52 = cos_15[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_15 = None
        sin_15 = emb_5.sin()
        emb_5 = None
        getitem_53 = sin_15[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_15 = None
        cos_16 = getitem_52[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_16 = getitem_53[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_40 = query_layer_16 * cos_16
        cos_16 = None
        chunk_10 = query_layer_16.chunk(2, dim=-1)
        query_layer_16 = None
        x1_10 = chunk_10[0]
        x2_10 = chunk_10[1]
        chunk_10 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_16 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_41 = cat_16 * sin_16
        cat_16 = sin_16 = None
        add_31 = mul_40 + mul_41
        mul_40 = mul_41 = None
        query_layer_17 = add_31.to(dtype=torch.float32)
        add_31 = None
        cos_17 = getitem_52[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_17 = getitem_53[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_42 = key_layer_10 * cos_17
        cos_17 = None
        chunk_11 = key_layer_10.chunk(2, dim=-1)
        key_layer_10 = None
        x1_11 = chunk_11[0]
        x2_11 = chunk_11[1]
        chunk_11 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_17 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_43 = cat_17 * sin_17
        cat_17 = sin_17 = None
        add_32 = mul_42 + mul_43
        mul_42 = mul_43 = None
        key_layer_11 = add_32.to(dtype=torch.float32)
        add_32 = None
        transpose_23 = key_layer_11.transpose(-1, -2)
        key_layer_11 = None
        attention_scores_10 = torch.matmul(query_layer_17, transpose_23)
        query_layer_17 = transpose_23 = None
        attention_scores_11 = attention_scores_10 + extended_attention_mask_2
        attention_scores_10 = None
        attention_probs_10 = torch.nn.functional.softmax(attention_scores_11, dim=-1)
        attention_scores_11 = None
        attention_probs_11 = torch.nn.functional.dropout(
            attention_probs_10, 0.0, False, False
        )
        attention_probs_10 = None
        to_26 = attention_probs_11.to(torch.float32)
        attention_probs_11 = None
        context_layer_15 = torch.matmul(to_26, value_layer_5)
        to_26 = value_layer_5 = None
        permute_5 = context_layer_15.permute(0, 2, 1, 3)
        context_layer_15 = None
        context_layer_16 = permute_5.contiguous()
        permute_5 = None
        context_layer_17 = context_layer_16.view((1, 13, 1280))
        context_layer_16 = None
        hidden_states_40 = torch._C._nn.linear(
            context_layer_17,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_17 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_41 = torch.nn.functional.dropout(
            hidden_states_40, 0.0, False, False
        )
        hidden_states_40 = None
        hidden_states_42 = hidden_states_41 + hidden_states_39
        hidden_states_41 = hidden_states_39 = None
        attention_output_ln_5 = torch.nn.functional.layer_norm(
            hidden_states_42,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_5_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_43 = torch._C._nn.linear(
            attention_output_ln_5,
            l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_5 = l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_44 = hidden_states_43 * 0.5
        truediv_7 = hidden_states_43 / 1.4142135623730951
        hidden_states_43 = None
        erf_5 = torch.erf(truediv_7)
        truediv_7 = None
        add_35 = 1.0 + erf_5
        erf_5 = None
        hidden_states_44 = mul_44 * add_35
        mul_44 = add_35 = None
        hidden_states_45 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_44 = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_46 = torch.nn.functional.dropout(
            hidden_states_45, 0.0, False, False
        )
        hidden_states_45 = None
        hidden_states_47 = hidden_states_46 + hidden_states_42
        hidden_states_46 = hidden_states_42 = None
        hidden_states_ln_6 = torch.nn.functional.layer_norm(
            hidden_states_47,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_36 = torch._C._nn.linear(
            hidden_states_ln_6,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_24 = linear_36.view((1, -1, 20, 64))
        linear_36 = None
        query_layer_18 = view_24.transpose(1, 2)
        view_24 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_ln_6,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_25 = linear_37.view((1, -1, 20, 64))
        linear_37 = None
        key_layer_12 = view_25.transpose(1, 2)
        view_25 = None
        linear_38 = torch._C._nn.linear(
            hidden_states_ln_6,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_6 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_26 = linear_38.view((1, -1, 20, 64))
        linear_38 = None
        value_layer_6 = view_26.transpose(1, 2)
        view_26 = None
        query_layer_19 = query_layer_18 * 0.125
        query_layer_18 = None
        arange_6 = torch.arange(13, device=device(type="cuda", index=0))
        t_6 = arange_6.type_as(
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_6 = None
        freqs_6 = torch.outer(
            t_6,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_6 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_18 = torch.cat((freqs_6, freqs_6), dim=-1)
        freqs_6 = None
        emb_6 = cat_18.to(device(type="cuda", index=0))
        cat_18 = None
        cos_18 = emb_6.cos()
        getitem_62 = cos_18[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_18 = None
        sin_18 = emb_6.sin()
        emb_6 = None
        getitem_63 = sin_18[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_18 = None
        cos_19 = getitem_62[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_19 = getitem_63[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_47 = query_layer_19 * cos_19
        cos_19 = None
        chunk_12 = query_layer_19.chunk(2, dim=-1)
        query_layer_19 = None
        x1_12 = chunk_12[0]
        x2_12 = chunk_12[1]
        chunk_12 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_19 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_48 = cat_19 * sin_19
        cat_19 = sin_19 = None
        add_37 = mul_47 + mul_48
        mul_47 = mul_48 = None
        query_layer_20 = add_37.to(dtype=torch.float32)
        add_37 = None
        cos_20 = getitem_62[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_20 = getitem_63[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_49 = key_layer_12 * cos_20
        cos_20 = None
        chunk_13 = key_layer_12.chunk(2, dim=-1)
        key_layer_12 = None
        x1_13 = chunk_13[0]
        x2_13 = chunk_13[1]
        chunk_13 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_20 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_50 = cat_20 * sin_20
        cat_20 = sin_20 = None
        add_38 = mul_49 + mul_50
        mul_49 = mul_50 = None
        key_layer_13 = add_38.to(dtype=torch.float32)
        add_38 = None
        transpose_27 = key_layer_13.transpose(-1, -2)
        key_layer_13 = None
        attention_scores_12 = torch.matmul(query_layer_20, transpose_27)
        query_layer_20 = transpose_27 = None
        attention_scores_13 = attention_scores_12 + extended_attention_mask_2
        attention_scores_12 = None
        attention_probs_12 = torch.nn.functional.softmax(attention_scores_13, dim=-1)
        attention_scores_13 = None
        attention_probs_13 = torch.nn.functional.dropout(
            attention_probs_12, 0.0, False, False
        )
        attention_probs_12 = None
        to_30 = attention_probs_13.to(torch.float32)
        attention_probs_13 = None
        context_layer_18 = torch.matmul(to_30, value_layer_6)
        to_30 = value_layer_6 = None
        permute_6 = context_layer_18.permute(0, 2, 1, 3)
        context_layer_18 = None
        context_layer_19 = permute_6.contiguous()
        permute_6 = None
        context_layer_20 = context_layer_19.view((1, 13, 1280))
        context_layer_19 = None
        hidden_states_48 = torch._C._nn.linear(
            context_layer_20,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_20 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_49 = torch.nn.functional.dropout(
            hidden_states_48, 0.0, False, False
        )
        hidden_states_48 = None
        hidden_states_50 = hidden_states_49 + hidden_states_47
        hidden_states_49 = hidden_states_47 = None
        attention_output_ln_6 = torch.nn.functional.layer_norm(
            hidden_states_50,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_6_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_51 = torch._C._nn.linear(
            attention_output_ln_6,
            l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_6 = l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_51 = hidden_states_51 * 0.5
        truediv_8 = hidden_states_51 / 1.4142135623730951
        hidden_states_51 = None
        erf_6 = torch.erf(truediv_8)
        truediv_8 = None
        add_41 = 1.0 + erf_6
        erf_6 = None
        hidden_states_52 = mul_51 * add_41
        mul_51 = add_41 = None
        hidden_states_53 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_52 = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_54 = torch.nn.functional.dropout(
            hidden_states_53, 0.0, False, False
        )
        hidden_states_53 = None
        hidden_states_55 = hidden_states_54 + hidden_states_50
        hidden_states_54 = hidden_states_50 = None
        hidden_states_ln_7 = torch.nn.functional.layer_norm(
            hidden_states_55,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_42 = torch._C._nn.linear(
            hidden_states_ln_7,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_28 = linear_42.view((1, -1, 20, 64))
        linear_42 = None
        query_layer_21 = view_28.transpose(1, 2)
        view_28 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_ln_7,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_29 = linear_43.view((1, -1, 20, 64))
        linear_43 = None
        key_layer_14 = view_29.transpose(1, 2)
        view_29 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_ln_7,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_7 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_30 = linear_44.view((1, -1, 20, 64))
        linear_44 = None
        value_layer_7 = view_30.transpose(1, 2)
        view_30 = None
        query_layer_22 = query_layer_21 * 0.125
        query_layer_21 = None
        arange_7 = torch.arange(13, device=device(type="cuda", index=0))
        t_7 = arange_7.type_as(
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_7 = None
        freqs_7 = torch.outer(
            t_7,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_7 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_21 = torch.cat((freqs_7, freqs_7), dim=-1)
        freqs_7 = None
        emb_7 = cat_21.to(device(type="cuda", index=0))
        cat_21 = None
        cos_21 = emb_7.cos()
        getitem_72 = cos_21[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_21 = None
        sin_21 = emb_7.sin()
        emb_7 = None
        getitem_73 = sin_21[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_21 = None
        cos_22 = getitem_72[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_22 = getitem_73[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_54 = query_layer_22 * cos_22
        cos_22 = None
        chunk_14 = query_layer_22.chunk(2, dim=-1)
        query_layer_22 = None
        x1_14 = chunk_14[0]
        x2_14 = chunk_14[1]
        chunk_14 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_22 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_55 = cat_22 * sin_22
        cat_22 = sin_22 = None
        add_43 = mul_54 + mul_55
        mul_54 = mul_55 = None
        query_layer_23 = add_43.to(dtype=torch.float32)
        add_43 = None
        cos_23 = getitem_72[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_23 = getitem_73[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_56 = key_layer_14 * cos_23
        cos_23 = None
        chunk_15 = key_layer_14.chunk(2, dim=-1)
        key_layer_14 = None
        x1_15 = chunk_15[0]
        x2_15 = chunk_15[1]
        chunk_15 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_23 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_57 = cat_23 * sin_23
        cat_23 = sin_23 = None
        add_44 = mul_56 + mul_57
        mul_56 = mul_57 = None
        key_layer_15 = add_44.to(dtype=torch.float32)
        add_44 = None
        transpose_31 = key_layer_15.transpose(-1, -2)
        key_layer_15 = None
        attention_scores_14 = torch.matmul(query_layer_23, transpose_31)
        query_layer_23 = transpose_31 = None
        attention_scores_15 = attention_scores_14 + extended_attention_mask_2
        attention_scores_14 = None
        attention_probs_14 = torch.nn.functional.softmax(attention_scores_15, dim=-1)
        attention_scores_15 = None
        attention_probs_15 = torch.nn.functional.dropout(
            attention_probs_14, 0.0, False, False
        )
        attention_probs_14 = None
        to_34 = attention_probs_15.to(torch.float32)
        attention_probs_15 = None
        context_layer_21 = torch.matmul(to_34, value_layer_7)
        to_34 = value_layer_7 = None
        permute_7 = context_layer_21.permute(0, 2, 1, 3)
        context_layer_21 = None
        context_layer_22 = permute_7.contiguous()
        permute_7 = None
        context_layer_23 = context_layer_22.view((1, 13, 1280))
        context_layer_22 = None
        hidden_states_56 = torch._C._nn.linear(
            context_layer_23,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_23 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_57 = torch.nn.functional.dropout(
            hidden_states_56, 0.0, False, False
        )
        hidden_states_56 = None
        hidden_states_58 = hidden_states_57 + hidden_states_55
        hidden_states_57 = hidden_states_55 = None
        attention_output_ln_7 = torch.nn.functional.layer_norm(
            hidden_states_58,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_7_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_59 = torch._C._nn.linear(
            attention_output_ln_7,
            l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_7 = l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_58 = hidden_states_59 * 0.5
        truediv_9 = hidden_states_59 / 1.4142135623730951
        hidden_states_59 = None
        erf_7 = torch.erf(truediv_9)
        truediv_9 = None
        add_47 = 1.0 + erf_7
        erf_7 = None
        hidden_states_60 = mul_58 * add_47
        mul_58 = add_47 = None
        hidden_states_61 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_60 = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_62 = torch.nn.functional.dropout(
            hidden_states_61, 0.0, False, False
        )
        hidden_states_61 = None
        hidden_states_63 = hidden_states_62 + hidden_states_58
        hidden_states_62 = hidden_states_58 = None
        hidden_states_ln_8 = torch.nn.functional.layer_norm(
            hidden_states_63,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            hidden_states_ln_8,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_32 = linear_48.view((1, -1, 20, 64))
        linear_48 = None
        query_layer_24 = view_32.transpose(1, 2)
        view_32 = None
        linear_49 = torch._C._nn.linear(
            hidden_states_ln_8,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_33 = linear_49.view((1, -1, 20, 64))
        linear_49 = None
        key_layer_16 = view_33.transpose(1, 2)
        view_33 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_ln_8,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_8 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_34 = linear_50.view((1, -1, 20, 64))
        linear_50 = None
        value_layer_8 = view_34.transpose(1, 2)
        view_34 = None
        query_layer_25 = query_layer_24 * 0.125
        query_layer_24 = None
        arange_8 = torch.arange(13, device=device(type="cuda", index=0))
        t_8 = arange_8.type_as(
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_8 = None
        freqs_8 = torch.outer(
            t_8,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_8 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_24 = torch.cat((freqs_8, freqs_8), dim=-1)
        freqs_8 = None
        emb_8 = cat_24.to(device(type="cuda", index=0))
        cat_24 = None
        cos_24 = emb_8.cos()
        getitem_82 = cos_24[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_24 = None
        sin_24 = emb_8.sin()
        emb_8 = None
        getitem_83 = sin_24[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_24 = None
        cos_25 = getitem_82[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_25 = getitem_83[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_61 = query_layer_25 * cos_25
        cos_25 = None
        chunk_16 = query_layer_25.chunk(2, dim=-1)
        query_layer_25 = None
        x1_16 = chunk_16[0]
        x2_16 = chunk_16[1]
        chunk_16 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_25 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_62 = cat_25 * sin_25
        cat_25 = sin_25 = None
        add_49 = mul_61 + mul_62
        mul_61 = mul_62 = None
        query_layer_26 = add_49.to(dtype=torch.float32)
        add_49 = None
        cos_26 = getitem_82[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_26 = getitem_83[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_63 = key_layer_16 * cos_26
        cos_26 = None
        chunk_17 = key_layer_16.chunk(2, dim=-1)
        key_layer_16 = None
        x1_17 = chunk_17[0]
        x2_17 = chunk_17[1]
        chunk_17 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_26 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_64 = cat_26 * sin_26
        cat_26 = sin_26 = None
        add_50 = mul_63 + mul_64
        mul_63 = mul_64 = None
        key_layer_17 = add_50.to(dtype=torch.float32)
        add_50 = None
        transpose_35 = key_layer_17.transpose(-1, -2)
        key_layer_17 = None
        attention_scores_16 = torch.matmul(query_layer_26, transpose_35)
        query_layer_26 = transpose_35 = None
        attention_scores_17 = attention_scores_16 + extended_attention_mask_2
        attention_scores_16 = None
        attention_probs_16 = torch.nn.functional.softmax(attention_scores_17, dim=-1)
        attention_scores_17 = None
        attention_probs_17 = torch.nn.functional.dropout(
            attention_probs_16, 0.0, False, False
        )
        attention_probs_16 = None
        to_38 = attention_probs_17.to(torch.float32)
        attention_probs_17 = None
        context_layer_24 = torch.matmul(to_38, value_layer_8)
        to_38 = value_layer_8 = None
        permute_8 = context_layer_24.permute(0, 2, 1, 3)
        context_layer_24 = None
        context_layer_25 = permute_8.contiguous()
        permute_8 = None
        context_layer_26 = context_layer_25.view((1, 13, 1280))
        context_layer_25 = None
        hidden_states_64 = torch._C._nn.linear(
            context_layer_26,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_26 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_65 = torch.nn.functional.dropout(
            hidden_states_64, 0.0, False, False
        )
        hidden_states_64 = None
        hidden_states_66 = hidden_states_65 + hidden_states_63
        hidden_states_65 = hidden_states_63 = None
        attention_output_ln_8 = torch.nn.functional.layer_norm(
            hidden_states_66,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_8_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_67 = torch._C._nn.linear(
            attention_output_ln_8,
            l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_8 = l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_65 = hidden_states_67 * 0.5
        truediv_10 = hidden_states_67 / 1.4142135623730951
        hidden_states_67 = None
        erf_8 = torch.erf(truediv_10)
        truediv_10 = None
        add_53 = 1.0 + erf_8
        erf_8 = None
        hidden_states_68 = mul_65 * add_53
        mul_65 = add_53 = None
        hidden_states_69 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_68 = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, 0.0, False, False
        )
        hidden_states_69 = None
        hidden_states_71 = hidden_states_70 + hidden_states_66
        hidden_states_70 = hidden_states_66 = None
        hidden_states_ln_9 = torch.nn.functional.layer_norm(
            hidden_states_71,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            hidden_states_ln_9,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_36 = linear_54.view((1, -1, 20, 64))
        linear_54 = None
        query_layer_27 = view_36.transpose(1, 2)
        view_36 = None
        linear_55 = torch._C._nn.linear(
            hidden_states_ln_9,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_37 = linear_55.view((1, -1, 20, 64))
        linear_55 = None
        key_layer_18 = view_37.transpose(1, 2)
        view_37 = None
        linear_56 = torch._C._nn.linear(
            hidden_states_ln_9,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_9 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_38 = linear_56.view((1, -1, 20, 64))
        linear_56 = None
        value_layer_9 = view_38.transpose(1, 2)
        view_38 = None
        query_layer_28 = query_layer_27 * 0.125
        query_layer_27 = None
        arange_9 = torch.arange(13, device=device(type="cuda", index=0))
        t_9 = arange_9.type_as(
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_9 = None
        freqs_9 = torch.outer(
            t_9,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_9 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_27 = torch.cat((freqs_9, freqs_9), dim=-1)
        freqs_9 = None
        emb_9 = cat_27.to(device(type="cuda", index=0))
        cat_27 = None
        cos_27 = emb_9.cos()
        getitem_92 = cos_27[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_27 = None
        sin_27 = emb_9.sin()
        emb_9 = None
        getitem_93 = sin_27[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_27 = None
        cos_28 = getitem_92[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_28 = getitem_93[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_68 = query_layer_28 * cos_28
        cos_28 = None
        chunk_18 = query_layer_28.chunk(2, dim=-1)
        query_layer_28 = None
        x1_18 = chunk_18[0]
        x2_18 = chunk_18[1]
        chunk_18 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_28 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_69 = cat_28 * sin_28
        cat_28 = sin_28 = None
        add_55 = mul_68 + mul_69
        mul_68 = mul_69 = None
        query_layer_29 = add_55.to(dtype=torch.float32)
        add_55 = None
        cos_29 = getitem_92[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_29 = getitem_93[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_70 = key_layer_18 * cos_29
        cos_29 = None
        chunk_19 = key_layer_18.chunk(2, dim=-1)
        key_layer_18 = None
        x1_19 = chunk_19[0]
        x2_19 = chunk_19[1]
        chunk_19 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_29 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_71 = cat_29 * sin_29
        cat_29 = sin_29 = None
        add_56 = mul_70 + mul_71
        mul_70 = mul_71 = None
        key_layer_19 = add_56.to(dtype=torch.float32)
        add_56 = None
        transpose_39 = key_layer_19.transpose(-1, -2)
        key_layer_19 = None
        attention_scores_18 = torch.matmul(query_layer_29, transpose_39)
        query_layer_29 = transpose_39 = None
        attention_scores_19 = attention_scores_18 + extended_attention_mask_2
        attention_scores_18 = None
        attention_probs_18 = torch.nn.functional.softmax(attention_scores_19, dim=-1)
        attention_scores_19 = None
        attention_probs_19 = torch.nn.functional.dropout(
            attention_probs_18, 0.0, False, False
        )
        attention_probs_18 = None
        to_42 = attention_probs_19.to(torch.float32)
        attention_probs_19 = None
        context_layer_27 = torch.matmul(to_42, value_layer_9)
        to_42 = value_layer_9 = None
        permute_9 = context_layer_27.permute(0, 2, 1, 3)
        context_layer_27 = None
        context_layer_28 = permute_9.contiguous()
        permute_9 = None
        context_layer_29 = context_layer_28.view((1, 13, 1280))
        context_layer_28 = None
        hidden_states_72 = torch._C._nn.linear(
            context_layer_29,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_29 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_73 = torch.nn.functional.dropout(
            hidden_states_72, 0.0, False, False
        )
        hidden_states_72 = None
        hidden_states_74 = hidden_states_73 + hidden_states_71
        hidden_states_73 = hidden_states_71 = None
        attention_output_ln_9 = torch.nn.functional.layer_norm(
            hidden_states_74,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_9_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_75 = torch._C._nn.linear(
            attention_output_ln_9,
            l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_9 = l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_72 = hidden_states_75 * 0.5
        truediv_11 = hidden_states_75 / 1.4142135623730951
        hidden_states_75 = None
        erf_9 = torch.erf(truediv_11)
        truediv_11 = None
        add_59 = 1.0 + erf_9
        erf_9 = None
        hidden_states_76 = mul_72 * add_59
        mul_72 = add_59 = None
        hidden_states_77 = torch._C._nn.linear(
            hidden_states_76,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_76 = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_78 = torch.nn.functional.dropout(
            hidden_states_77, 0.0, False, False
        )
        hidden_states_77 = None
        hidden_states_79 = hidden_states_78 + hidden_states_74
        hidden_states_78 = hidden_states_74 = None
        hidden_states_ln_10 = torch.nn.functional.layer_norm(
            hidden_states_79,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            hidden_states_ln_10,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_40 = linear_60.view((1, -1, 20, 64))
        linear_60 = None
        query_layer_30 = view_40.transpose(1, 2)
        view_40 = None
        linear_61 = torch._C._nn.linear(
            hidden_states_ln_10,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_41 = linear_61.view((1, -1, 20, 64))
        linear_61 = None
        key_layer_20 = view_41.transpose(1, 2)
        view_41 = None
        linear_62 = torch._C._nn.linear(
            hidden_states_ln_10,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_10 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_42 = linear_62.view((1, -1, 20, 64))
        linear_62 = None
        value_layer_10 = view_42.transpose(1, 2)
        view_42 = None
        query_layer_31 = query_layer_30 * 0.125
        query_layer_30 = None
        arange_10 = torch.arange(13, device=device(type="cuda", index=0))
        t_10 = arange_10.type_as(
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_10 = None
        freqs_10 = torch.outer(
            t_10,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_10 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_30 = torch.cat((freqs_10, freqs_10), dim=-1)
        freqs_10 = None
        emb_10 = cat_30.to(device(type="cuda", index=0))
        cat_30 = None
        cos_30 = emb_10.cos()
        getitem_102 = cos_30[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_30 = None
        sin_30 = emb_10.sin()
        emb_10 = None
        getitem_103 = sin_30[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_30 = None
        cos_31 = getitem_102[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_31 = getitem_103[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_75 = query_layer_31 * cos_31
        cos_31 = None
        chunk_20 = query_layer_31.chunk(2, dim=-1)
        query_layer_31 = None
        x1_20 = chunk_20[0]
        x2_20 = chunk_20[1]
        chunk_20 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_31 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_76 = cat_31 * sin_31
        cat_31 = sin_31 = None
        add_61 = mul_75 + mul_76
        mul_75 = mul_76 = None
        query_layer_32 = add_61.to(dtype=torch.float32)
        add_61 = None
        cos_32 = getitem_102[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_32 = getitem_103[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_77 = key_layer_20 * cos_32
        cos_32 = None
        chunk_21 = key_layer_20.chunk(2, dim=-1)
        key_layer_20 = None
        x1_21 = chunk_21[0]
        x2_21 = chunk_21[1]
        chunk_21 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_32 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_78 = cat_32 * sin_32
        cat_32 = sin_32 = None
        add_62 = mul_77 + mul_78
        mul_77 = mul_78 = None
        key_layer_21 = add_62.to(dtype=torch.float32)
        add_62 = None
        transpose_43 = key_layer_21.transpose(-1, -2)
        key_layer_21 = None
        attention_scores_20 = torch.matmul(query_layer_32, transpose_43)
        query_layer_32 = transpose_43 = None
        attention_scores_21 = attention_scores_20 + extended_attention_mask_2
        attention_scores_20 = None
        attention_probs_20 = torch.nn.functional.softmax(attention_scores_21, dim=-1)
        attention_scores_21 = None
        attention_probs_21 = torch.nn.functional.dropout(
            attention_probs_20, 0.0, False, False
        )
        attention_probs_20 = None
        to_46 = attention_probs_21.to(torch.float32)
        attention_probs_21 = None
        context_layer_30 = torch.matmul(to_46, value_layer_10)
        to_46 = value_layer_10 = None
        permute_10 = context_layer_30.permute(0, 2, 1, 3)
        context_layer_30 = None
        context_layer_31 = permute_10.contiguous()
        permute_10 = None
        context_layer_32 = context_layer_31.view((1, 13, 1280))
        context_layer_31 = None
        hidden_states_80 = torch._C._nn.linear(
            context_layer_32,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_32 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_81 = torch.nn.functional.dropout(
            hidden_states_80, 0.0, False, False
        )
        hidden_states_80 = None
        hidden_states_82 = hidden_states_81 + hidden_states_79
        hidden_states_81 = hidden_states_79 = None
        attention_output_ln_10 = torch.nn.functional.layer_norm(
            hidden_states_82,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_10_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_83 = torch._C._nn.linear(
            attention_output_ln_10,
            l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_10 = l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_79 = hidden_states_83 * 0.5
        truediv_12 = hidden_states_83 / 1.4142135623730951
        hidden_states_83 = None
        erf_10 = torch.erf(truediv_12)
        truediv_12 = None
        add_65 = 1.0 + erf_10
        erf_10 = None
        hidden_states_84 = mul_79 * add_65
        mul_79 = add_65 = None
        hidden_states_85 = torch._C._nn.linear(
            hidden_states_84,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_84 = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_86 = torch.nn.functional.dropout(
            hidden_states_85, 0.0, False, False
        )
        hidden_states_85 = None
        hidden_states_87 = hidden_states_86 + hidden_states_82
        hidden_states_86 = hidden_states_82 = None
        hidden_states_ln_11 = torch.nn.functional.layer_norm(
            hidden_states_87,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_66 = torch._C._nn.linear(
            hidden_states_ln_11,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_44 = linear_66.view((1, -1, 20, 64))
        linear_66 = None
        query_layer_33 = view_44.transpose(1, 2)
        view_44 = None
        linear_67 = torch._C._nn.linear(
            hidden_states_ln_11,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_45 = linear_67.view((1, -1, 20, 64))
        linear_67 = None
        key_layer_22 = view_45.transpose(1, 2)
        view_45 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_ln_11,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_11 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_46 = linear_68.view((1, -1, 20, 64))
        linear_68 = None
        value_layer_11 = view_46.transpose(1, 2)
        view_46 = None
        query_layer_34 = query_layer_33 * 0.125
        query_layer_33 = None
        arange_11 = torch.arange(13, device=device(type="cuda", index=0))
        t_11 = arange_11.type_as(
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_11 = None
        freqs_11 = torch.outer(
            t_11,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_11 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_33 = torch.cat((freqs_11, freqs_11), dim=-1)
        freqs_11 = None
        emb_11 = cat_33.to(device(type="cuda", index=0))
        cat_33 = None
        cos_33 = emb_11.cos()
        getitem_112 = cos_33[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_33 = None
        sin_33 = emb_11.sin()
        emb_11 = None
        getitem_113 = sin_33[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_33 = None
        cos_34 = getitem_112[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_34 = getitem_113[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_82 = query_layer_34 * cos_34
        cos_34 = None
        chunk_22 = query_layer_34.chunk(2, dim=-1)
        query_layer_34 = None
        x1_22 = chunk_22[0]
        x2_22 = chunk_22[1]
        chunk_22 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_34 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_83 = cat_34 * sin_34
        cat_34 = sin_34 = None
        add_67 = mul_82 + mul_83
        mul_82 = mul_83 = None
        query_layer_35 = add_67.to(dtype=torch.float32)
        add_67 = None
        cos_35 = getitem_112[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_35 = getitem_113[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_84 = key_layer_22 * cos_35
        cos_35 = None
        chunk_23 = key_layer_22.chunk(2, dim=-1)
        key_layer_22 = None
        x1_23 = chunk_23[0]
        x2_23 = chunk_23[1]
        chunk_23 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_35 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_85 = cat_35 * sin_35
        cat_35 = sin_35 = None
        add_68 = mul_84 + mul_85
        mul_84 = mul_85 = None
        key_layer_23 = add_68.to(dtype=torch.float32)
        add_68 = None
        transpose_47 = key_layer_23.transpose(-1, -2)
        key_layer_23 = None
        attention_scores_22 = torch.matmul(query_layer_35, transpose_47)
        query_layer_35 = transpose_47 = None
        attention_scores_23 = attention_scores_22 + extended_attention_mask_2
        attention_scores_22 = None
        attention_probs_22 = torch.nn.functional.softmax(attention_scores_23, dim=-1)
        attention_scores_23 = None
        attention_probs_23 = torch.nn.functional.dropout(
            attention_probs_22, 0.0, False, False
        )
        attention_probs_22 = None
        to_50 = attention_probs_23.to(torch.float32)
        attention_probs_23 = None
        context_layer_33 = torch.matmul(to_50, value_layer_11)
        to_50 = value_layer_11 = None
        permute_11 = context_layer_33.permute(0, 2, 1, 3)
        context_layer_33 = None
        context_layer_34 = permute_11.contiguous()
        permute_11 = None
        context_layer_35 = context_layer_34.view((1, 13, 1280))
        context_layer_34 = None
        hidden_states_88 = torch._C._nn.linear(
            context_layer_35,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_35 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_89 = torch.nn.functional.dropout(
            hidden_states_88, 0.0, False, False
        )
        hidden_states_88 = None
        hidden_states_90 = hidden_states_89 + hidden_states_87
        hidden_states_89 = hidden_states_87 = None
        attention_output_ln_11 = torch.nn.functional.layer_norm(
            hidden_states_90,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_11_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_91 = torch._C._nn.linear(
            attention_output_ln_11,
            l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_11 = l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_86 = hidden_states_91 * 0.5
        truediv_13 = hidden_states_91 / 1.4142135623730951
        hidden_states_91 = None
        erf_11 = torch.erf(truediv_13)
        truediv_13 = None
        add_71 = 1.0 + erf_11
        erf_11 = None
        hidden_states_92 = mul_86 * add_71
        mul_86 = add_71 = None
        hidden_states_93 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_94 = torch.nn.functional.dropout(
            hidden_states_93, 0.0, False, False
        )
        hidden_states_93 = None
        hidden_states_95 = hidden_states_94 + hidden_states_90
        hidden_states_94 = hidden_states_90 = None
        hidden_states_ln_12 = torch.nn.functional.layer_norm(
            hidden_states_95,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_72 = torch._C._nn.linear(
            hidden_states_ln_12,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_48 = linear_72.view((1, -1, 20, 64))
        linear_72 = None
        query_layer_36 = view_48.transpose(1, 2)
        view_48 = None
        linear_73 = torch._C._nn.linear(
            hidden_states_ln_12,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_49 = linear_73.view((1, -1, 20, 64))
        linear_73 = None
        key_layer_24 = view_49.transpose(1, 2)
        view_49 = None
        linear_74 = torch._C._nn.linear(
            hidden_states_ln_12,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_12 = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_50 = linear_74.view((1, -1, 20, 64))
        linear_74 = None
        value_layer_12 = view_50.transpose(1, 2)
        view_50 = None
        query_layer_37 = query_layer_36 * 0.125
        query_layer_36 = None
        arange_12 = torch.arange(13, device=device(type="cuda", index=0))
        t_12 = arange_12.type_as(
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_12 = None
        freqs_12 = torch.outer(
            t_12,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_12 = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_36 = torch.cat((freqs_12, freqs_12), dim=-1)
        freqs_12 = None
        emb_12 = cat_36.to(device(type="cuda", index=0))
        cat_36 = None
        cos_36 = emb_12.cos()
        getitem_122 = cos_36[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_36 = None
        sin_36 = emb_12.sin()
        emb_12 = None
        getitem_123 = sin_36[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_36 = None
        cos_37 = getitem_122[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_37 = getitem_123[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_89 = query_layer_37 * cos_37
        cos_37 = None
        chunk_24 = query_layer_37.chunk(2, dim=-1)
        query_layer_37 = None
        x1_24 = chunk_24[0]
        x2_24 = chunk_24[1]
        chunk_24 = None
        neg_24 = -x2_24
        x2_24 = None
        cat_37 = torch.cat((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        mul_90 = cat_37 * sin_37
        cat_37 = sin_37 = None
        add_73 = mul_89 + mul_90
        mul_89 = mul_90 = None
        query_layer_38 = add_73.to(dtype=torch.float32)
        add_73 = None
        cos_38 = getitem_122[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_38 = getitem_123[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_91 = key_layer_24 * cos_38
        cos_38 = None
        chunk_25 = key_layer_24.chunk(2, dim=-1)
        key_layer_24 = None
        x1_25 = chunk_25[0]
        x2_25 = chunk_25[1]
        chunk_25 = None
        neg_25 = -x2_25
        x2_25 = None
        cat_38 = torch.cat((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        mul_92 = cat_38 * sin_38
        cat_38 = sin_38 = None
        add_74 = mul_91 + mul_92
        mul_91 = mul_92 = None
        key_layer_25 = add_74.to(dtype=torch.float32)
        add_74 = None
        transpose_51 = key_layer_25.transpose(-1, -2)
        key_layer_25 = None
        attention_scores_24 = torch.matmul(query_layer_38, transpose_51)
        query_layer_38 = transpose_51 = None
        attention_scores_25 = attention_scores_24 + extended_attention_mask_2
        attention_scores_24 = None
        attention_probs_24 = torch.nn.functional.softmax(attention_scores_25, dim=-1)
        attention_scores_25 = None
        attention_probs_25 = torch.nn.functional.dropout(
            attention_probs_24, 0.0, False, False
        )
        attention_probs_24 = None
        to_54 = attention_probs_25.to(torch.float32)
        attention_probs_25 = None
        context_layer_36 = torch.matmul(to_54, value_layer_12)
        to_54 = value_layer_12 = None
        permute_12 = context_layer_36.permute(0, 2, 1, 3)
        context_layer_36 = None
        context_layer_37 = permute_12.contiguous()
        permute_12 = None
        context_layer_38 = context_layer_37.view((1, 13, 1280))
        context_layer_37 = None
        hidden_states_96 = torch._C._nn.linear(
            context_layer_38,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_38 = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_97 = torch.nn.functional.dropout(
            hidden_states_96, 0.0, False, False
        )
        hidden_states_96 = None
        hidden_states_98 = hidden_states_97 + hidden_states_95
        hidden_states_97 = hidden_states_95 = None
        attention_output_ln_12 = torch.nn.functional.layer_norm(
            hidden_states_98,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_12_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_99 = torch._C._nn.linear(
            attention_output_ln_12,
            l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_12 = l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_93 = hidden_states_99 * 0.5
        truediv_14 = hidden_states_99 / 1.4142135623730951
        hidden_states_99 = None
        erf_12 = torch.erf(truediv_14)
        truediv_14 = None
        add_77 = 1.0 + erf_12
        erf_12 = None
        hidden_states_100 = mul_93 * add_77
        mul_93 = add_77 = None
        hidden_states_101 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_100 = l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_102 = torch.nn.functional.dropout(
            hidden_states_101, 0.0, False, False
        )
        hidden_states_101 = None
        hidden_states_103 = hidden_states_102 + hidden_states_98
        hidden_states_102 = hidden_states_98 = None
        hidden_states_ln_13 = torch.nn.functional.layer_norm(
            hidden_states_103,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_78 = torch._C._nn.linear(
            hidden_states_ln_13,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_52 = linear_78.view((1, -1, 20, 64))
        linear_78 = None
        query_layer_39 = view_52.transpose(1, 2)
        view_52 = None
        linear_79 = torch._C._nn.linear(
            hidden_states_ln_13,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_53 = linear_79.view((1, -1, 20, 64))
        linear_79 = None
        key_layer_26 = view_53.transpose(1, 2)
        view_53 = None
        linear_80 = torch._C._nn.linear(
            hidden_states_ln_13,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_13 = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_54 = linear_80.view((1, -1, 20, 64))
        linear_80 = None
        value_layer_13 = view_54.transpose(1, 2)
        view_54 = None
        query_layer_40 = query_layer_39 * 0.125
        query_layer_39 = None
        arange_13 = torch.arange(13, device=device(type="cuda", index=0))
        t_13 = arange_13.type_as(
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_13 = None
        freqs_13 = torch.outer(
            t_13,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_13 = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_39 = torch.cat((freqs_13, freqs_13), dim=-1)
        freqs_13 = None
        emb_13 = cat_39.to(device(type="cuda", index=0))
        cat_39 = None
        cos_39 = emb_13.cos()
        getitem_132 = cos_39[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_39 = None
        sin_39 = emb_13.sin()
        emb_13 = None
        getitem_133 = sin_39[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_39 = None
        cos_40 = getitem_132[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_40 = getitem_133[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_96 = query_layer_40 * cos_40
        cos_40 = None
        chunk_26 = query_layer_40.chunk(2, dim=-1)
        query_layer_40 = None
        x1_26 = chunk_26[0]
        x2_26 = chunk_26[1]
        chunk_26 = None
        neg_26 = -x2_26
        x2_26 = None
        cat_40 = torch.cat((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        mul_97 = cat_40 * sin_40
        cat_40 = sin_40 = None
        add_79 = mul_96 + mul_97
        mul_96 = mul_97 = None
        query_layer_41 = add_79.to(dtype=torch.float32)
        add_79 = None
        cos_41 = getitem_132[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_41 = getitem_133[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_98 = key_layer_26 * cos_41
        cos_41 = None
        chunk_27 = key_layer_26.chunk(2, dim=-1)
        key_layer_26 = None
        x1_27 = chunk_27[0]
        x2_27 = chunk_27[1]
        chunk_27 = None
        neg_27 = -x2_27
        x2_27 = None
        cat_41 = torch.cat((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        mul_99 = cat_41 * sin_41
        cat_41 = sin_41 = None
        add_80 = mul_98 + mul_99
        mul_98 = mul_99 = None
        key_layer_27 = add_80.to(dtype=torch.float32)
        add_80 = None
        transpose_55 = key_layer_27.transpose(-1, -2)
        key_layer_27 = None
        attention_scores_26 = torch.matmul(query_layer_41, transpose_55)
        query_layer_41 = transpose_55 = None
        attention_scores_27 = attention_scores_26 + extended_attention_mask_2
        attention_scores_26 = None
        attention_probs_26 = torch.nn.functional.softmax(attention_scores_27, dim=-1)
        attention_scores_27 = None
        attention_probs_27 = torch.nn.functional.dropout(
            attention_probs_26, 0.0, False, False
        )
        attention_probs_26 = None
        to_58 = attention_probs_27.to(torch.float32)
        attention_probs_27 = None
        context_layer_39 = torch.matmul(to_58, value_layer_13)
        to_58 = value_layer_13 = None
        permute_13 = context_layer_39.permute(0, 2, 1, 3)
        context_layer_39 = None
        context_layer_40 = permute_13.contiguous()
        permute_13 = None
        context_layer_41 = context_layer_40.view((1, 13, 1280))
        context_layer_40 = None
        hidden_states_104 = torch._C._nn.linear(
            context_layer_41,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_41 = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_105 = torch.nn.functional.dropout(
            hidden_states_104, 0.0, False, False
        )
        hidden_states_104 = None
        hidden_states_106 = hidden_states_105 + hidden_states_103
        hidden_states_105 = hidden_states_103 = None
        attention_output_ln_13 = torch.nn.functional.layer_norm(
            hidden_states_106,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_13_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_107 = torch._C._nn.linear(
            attention_output_ln_13,
            l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_13 = l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_100 = hidden_states_107 * 0.5
        truediv_15 = hidden_states_107 / 1.4142135623730951
        hidden_states_107 = None
        erf_13 = torch.erf(truediv_15)
        truediv_15 = None
        add_83 = 1.0 + erf_13
        erf_13 = None
        hidden_states_108 = mul_100 * add_83
        mul_100 = add_83 = None
        hidden_states_109 = torch._C._nn.linear(
            hidden_states_108,
            l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_108 = l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_110 = torch.nn.functional.dropout(
            hidden_states_109, 0.0, False, False
        )
        hidden_states_109 = None
        hidden_states_111 = hidden_states_110 + hidden_states_106
        hidden_states_110 = hidden_states_106 = None
        hidden_states_ln_14 = torch.nn.functional.layer_norm(
            hidden_states_111,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            hidden_states_ln_14,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_56 = linear_84.view((1, -1, 20, 64))
        linear_84 = None
        query_layer_42 = view_56.transpose(1, 2)
        view_56 = None
        linear_85 = torch._C._nn.linear(
            hidden_states_ln_14,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_57 = linear_85.view((1, -1, 20, 64))
        linear_85 = None
        key_layer_28 = view_57.transpose(1, 2)
        view_57 = None
        linear_86 = torch._C._nn.linear(
            hidden_states_ln_14,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_14 = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_58 = linear_86.view((1, -1, 20, 64))
        linear_86 = None
        value_layer_14 = view_58.transpose(1, 2)
        view_58 = None
        query_layer_43 = query_layer_42 * 0.125
        query_layer_42 = None
        arange_14 = torch.arange(13, device=device(type="cuda", index=0))
        t_14 = arange_14.type_as(
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_14 = None
        freqs_14 = torch.outer(
            t_14,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_14 = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_42 = torch.cat((freqs_14, freqs_14), dim=-1)
        freqs_14 = None
        emb_14 = cat_42.to(device(type="cuda", index=0))
        cat_42 = None
        cos_42 = emb_14.cos()
        getitem_142 = cos_42[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_42 = None
        sin_42 = emb_14.sin()
        emb_14 = None
        getitem_143 = sin_42[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_42 = None
        cos_43 = getitem_142[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_43 = getitem_143[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_103 = query_layer_43 * cos_43
        cos_43 = None
        chunk_28 = query_layer_43.chunk(2, dim=-1)
        query_layer_43 = None
        x1_28 = chunk_28[0]
        x2_28 = chunk_28[1]
        chunk_28 = None
        neg_28 = -x2_28
        x2_28 = None
        cat_43 = torch.cat((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        mul_104 = cat_43 * sin_43
        cat_43 = sin_43 = None
        add_85 = mul_103 + mul_104
        mul_103 = mul_104 = None
        query_layer_44 = add_85.to(dtype=torch.float32)
        add_85 = None
        cos_44 = getitem_142[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_44 = getitem_143[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_105 = key_layer_28 * cos_44
        cos_44 = None
        chunk_29 = key_layer_28.chunk(2, dim=-1)
        key_layer_28 = None
        x1_29 = chunk_29[0]
        x2_29 = chunk_29[1]
        chunk_29 = None
        neg_29 = -x2_29
        x2_29 = None
        cat_44 = torch.cat((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        mul_106 = cat_44 * sin_44
        cat_44 = sin_44 = None
        add_86 = mul_105 + mul_106
        mul_105 = mul_106 = None
        key_layer_29 = add_86.to(dtype=torch.float32)
        add_86 = None
        transpose_59 = key_layer_29.transpose(-1, -2)
        key_layer_29 = None
        attention_scores_28 = torch.matmul(query_layer_44, transpose_59)
        query_layer_44 = transpose_59 = None
        attention_scores_29 = attention_scores_28 + extended_attention_mask_2
        attention_scores_28 = None
        attention_probs_28 = torch.nn.functional.softmax(attention_scores_29, dim=-1)
        attention_scores_29 = None
        attention_probs_29 = torch.nn.functional.dropout(
            attention_probs_28, 0.0, False, False
        )
        attention_probs_28 = None
        to_62 = attention_probs_29.to(torch.float32)
        attention_probs_29 = None
        context_layer_42 = torch.matmul(to_62, value_layer_14)
        to_62 = value_layer_14 = None
        permute_14 = context_layer_42.permute(0, 2, 1, 3)
        context_layer_42 = None
        context_layer_43 = permute_14.contiguous()
        permute_14 = None
        context_layer_44 = context_layer_43.view((1, 13, 1280))
        context_layer_43 = None
        hidden_states_112 = torch._C._nn.linear(
            context_layer_44,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_44 = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_113 = torch.nn.functional.dropout(
            hidden_states_112, 0.0, False, False
        )
        hidden_states_112 = None
        hidden_states_114 = hidden_states_113 + hidden_states_111
        hidden_states_113 = hidden_states_111 = None
        attention_output_ln_14 = torch.nn.functional.layer_norm(
            hidden_states_114,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_14_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_115 = torch._C._nn.linear(
            attention_output_ln_14,
            l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_14 = l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_107 = hidden_states_115 * 0.5
        truediv_16 = hidden_states_115 / 1.4142135623730951
        hidden_states_115 = None
        erf_14 = torch.erf(truediv_16)
        truediv_16 = None
        add_89 = 1.0 + erf_14
        erf_14 = None
        hidden_states_116 = mul_107 * add_89
        mul_107 = add_89 = None
        hidden_states_117 = torch._C._nn.linear(
            hidden_states_116,
            l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_116 = l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_118 = torch.nn.functional.dropout(
            hidden_states_117, 0.0, False, False
        )
        hidden_states_117 = None
        hidden_states_119 = hidden_states_118 + hidden_states_114
        hidden_states_118 = hidden_states_114 = None
        hidden_states_ln_15 = torch.nn.functional.layer_norm(
            hidden_states_119,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_90 = torch._C._nn.linear(
            hidden_states_ln_15,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_60 = linear_90.view((1, -1, 20, 64))
        linear_90 = None
        query_layer_45 = view_60.transpose(1, 2)
        view_60 = None
        linear_91 = torch._C._nn.linear(
            hidden_states_ln_15,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_61 = linear_91.view((1, -1, 20, 64))
        linear_91 = None
        key_layer_30 = view_61.transpose(1, 2)
        view_61 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_ln_15,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_15 = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_62 = linear_92.view((1, -1, 20, 64))
        linear_92 = None
        value_layer_15 = view_62.transpose(1, 2)
        view_62 = None
        query_layer_46 = query_layer_45 * 0.125
        query_layer_45 = None
        arange_15 = torch.arange(13, device=device(type="cuda", index=0))
        t_15 = arange_15.type_as(
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_15 = None
        freqs_15 = torch.outer(
            t_15,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_15 = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_45 = torch.cat((freqs_15, freqs_15), dim=-1)
        freqs_15 = None
        emb_15 = cat_45.to(device(type="cuda", index=0))
        cat_45 = None
        cos_45 = emb_15.cos()
        getitem_152 = cos_45[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_45 = None
        sin_45 = emb_15.sin()
        emb_15 = None
        getitem_153 = sin_45[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_45 = None
        cos_46 = getitem_152[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_46 = getitem_153[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_110 = query_layer_46 * cos_46
        cos_46 = None
        chunk_30 = query_layer_46.chunk(2, dim=-1)
        query_layer_46 = None
        x1_30 = chunk_30[0]
        x2_30 = chunk_30[1]
        chunk_30 = None
        neg_30 = -x2_30
        x2_30 = None
        cat_46 = torch.cat((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        mul_111 = cat_46 * sin_46
        cat_46 = sin_46 = None
        add_91 = mul_110 + mul_111
        mul_110 = mul_111 = None
        query_layer_47 = add_91.to(dtype=torch.float32)
        add_91 = None
        cos_47 = getitem_152[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_47 = getitem_153[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_112 = key_layer_30 * cos_47
        cos_47 = None
        chunk_31 = key_layer_30.chunk(2, dim=-1)
        key_layer_30 = None
        x1_31 = chunk_31[0]
        x2_31 = chunk_31[1]
        chunk_31 = None
        neg_31 = -x2_31
        x2_31 = None
        cat_47 = torch.cat((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        mul_113 = cat_47 * sin_47
        cat_47 = sin_47 = None
        add_92 = mul_112 + mul_113
        mul_112 = mul_113 = None
        key_layer_31 = add_92.to(dtype=torch.float32)
        add_92 = None
        transpose_63 = key_layer_31.transpose(-1, -2)
        key_layer_31 = None
        attention_scores_30 = torch.matmul(query_layer_47, transpose_63)
        query_layer_47 = transpose_63 = None
        attention_scores_31 = attention_scores_30 + extended_attention_mask_2
        attention_scores_30 = None
        attention_probs_30 = torch.nn.functional.softmax(attention_scores_31, dim=-1)
        attention_scores_31 = None
        attention_probs_31 = torch.nn.functional.dropout(
            attention_probs_30, 0.0, False, False
        )
        attention_probs_30 = None
        to_66 = attention_probs_31.to(torch.float32)
        attention_probs_31 = None
        context_layer_45 = torch.matmul(to_66, value_layer_15)
        to_66 = value_layer_15 = None
        permute_15 = context_layer_45.permute(0, 2, 1, 3)
        context_layer_45 = None
        context_layer_46 = permute_15.contiguous()
        permute_15 = None
        context_layer_47 = context_layer_46.view((1, 13, 1280))
        context_layer_46 = None
        hidden_states_120 = torch._C._nn.linear(
            context_layer_47,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_47 = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_121 = torch.nn.functional.dropout(
            hidden_states_120, 0.0, False, False
        )
        hidden_states_120 = None
        hidden_states_122 = hidden_states_121 + hidden_states_119
        hidden_states_121 = hidden_states_119 = None
        attention_output_ln_15 = torch.nn.functional.layer_norm(
            hidden_states_122,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_15_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_123 = torch._C._nn.linear(
            attention_output_ln_15,
            l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_15 = l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_114 = hidden_states_123 * 0.5
        truediv_17 = hidden_states_123 / 1.4142135623730951
        hidden_states_123 = None
        erf_15 = torch.erf(truediv_17)
        truediv_17 = None
        add_95 = 1.0 + erf_15
        erf_15 = None
        hidden_states_124 = mul_114 * add_95
        mul_114 = add_95 = None
        hidden_states_125 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_124 = l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_126 = torch.nn.functional.dropout(
            hidden_states_125, 0.0, False, False
        )
        hidden_states_125 = None
        hidden_states_127 = hidden_states_126 + hidden_states_122
        hidden_states_126 = hidden_states_122 = None
        hidden_states_ln_16 = torch.nn.functional.layer_norm(
            hidden_states_127,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_96 = torch._C._nn.linear(
            hidden_states_ln_16,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_64 = linear_96.view((1, -1, 20, 64))
        linear_96 = None
        query_layer_48 = view_64.transpose(1, 2)
        view_64 = None
        linear_97 = torch._C._nn.linear(
            hidden_states_ln_16,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_65 = linear_97.view((1, -1, 20, 64))
        linear_97 = None
        key_layer_32 = view_65.transpose(1, 2)
        view_65 = None
        linear_98 = torch._C._nn.linear(
            hidden_states_ln_16,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_16 = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_66 = linear_98.view((1, -1, 20, 64))
        linear_98 = None
        value_layer_16 = view_66.transpose(1, 2)
        view_66 = None
        query_layer_49 = query_layer_48 * 0.125
        query_layer_48 = None
        arange_16 = torch.arange(13, device=device(type="cuda", index=0))
        t_16 = arange_16.type_as(
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_16 = None
        freqs_16 = torch.outer(
            t_16,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_16 = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_48 = torch.cat((freqs_16, freqs_16), dim=-1)
        freqs_16 = None
        emb_16 = cat_48.to(device(type="cuda", index=0))
        cat_48 = None
        cos_48 = emb_16.cos()
        getitem_162 = cos_48[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_48 = None
        sin_48 = emb_16.sin()
        emb_16 = None
        getitem_163 = sin_48[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_48 = None
        cos_49 = getitem_162[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_49 = getitem_163[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_117 = query_layer_49 * cos_49
        cos_49 = None
        chunk_32 = query_layer_49.chunk(2, dim=-1)
        query_layer_49 = None
        x1_32 = chunk_32[0]
        x2_32 = chunk_32[1]
        chunk_32 = None
        neg_32 = -x2_32
        x2_32 = None
        cat_49 = torch.cat((neg_32, x1_32), dim=-1)
        neg_32 = x1_32 = None
        mul_118 = cat_49 * sin_49
        cat_49 = sin_49 = None
        add_97 = mul_117 + mul_118
        mul_117 = mul_118 = None
        query_layer_50 = add_97.to(dtype=torch.float32)
        add_97 = None
        cos_50 = getitem_162[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_50 = getitem_163[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_119 = key_layer_32 * cos_50
        cos_50 = None
        chunk_33 = key_layer_32.chunk(2, dim=-1)
        key_layer_32 = None
        x1_33 = chunk_33[0]
        x2_33 = chunk_33[1]
        chunk_33 = None
        neg_33 = -x2_33
        x2_33 = None
        cat_50 = torch.cat((neg_33, x1_33), dim=-1)
        neg_33 = x1_33 = None
        mul_120 = cat_50 * sin_50
        cat_50 = sin_50 = None
        add_98 = mul_119 + mul_120
        mul_119 = mul_120 = None
        key_layer_33 = add_98.to(dtype=torch.float32)
        add_98 = None
        transpose_67 = key_layer_33.transpose(-1, -2)
        key_layer_33 = None
        attention_scores_32 = torch.matmul(query_layer_50, transpose_67)
        query_layer_50 = transpose_67 = None
        attention_scores_33 = attention_scores_32 + extended_attention_mask_2
        attention_scores_32 = None
        attention_probs_32 = torch.nn.functional.softmax(attention_scores_33, dim=-1)
        attention_scores_33 = None
        attention_probs_33 = torch.nn.functional.dropout(
            attention_probs_32, 0.0, False, False
        )
        attention_probs_32 = None
        to_70 = attention_probs_33.to(torch.float32)
        attention_probs_33 = None
        context_layer_48 = torch.matmul(to_70, value_layer_16)
        to_70 = value_layer_16 = None
        permute_16 = context_layer_48.permute(0, 2, 1, 3)
        context_layer_48 = None
        context_layer_49 = permute_16.contiguous()
        permute_16 = None
        context_layer_50 = context_layer_49.view((1, 13, 1280))
        context_layer_49 = None
        hidden_states_128 = torch._C._nn.linear(
            context_layer_50,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_50 = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_129 = torch.nn.functional.dropout(
            hidden_states_128, 0.0, False, False
        )
        hidden_states_128 = None
        hidden_states_130 = hidden_states_129 + hidden_states_127
        hidden_states_129 = hidden_states_127 = None
        attention_output_ln_16 = torch.nn.functional.layer_norm(
            hidden_states_130,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_16_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_131 = torch._C._nn.linear(
            attention_output_ln_16,
            l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_16 = l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_121 = hidden_states_131 * 0.5
        truediv_18 = hidden_states_131 / 1.4142135623730951
        hidden_states_131 = None
        erf_16 = torch.erf(truediv_18)
        truediv_18 = None
        add_101 = 1.0 + erf_16
        erf_16 = None
        hidden_states_132 = mul_121 * add_101
        mul_121 = add_101 = None
        hidden_states_133 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_132 = l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_134 = torch.nn.functional.dropout(
            hidden_states_133, 0.0, False, False
        )
        hidden_states_133 = None
        hidden_states_135 = hidden_states_134 + hidden_states_130
        hidden_states_134 = hidden_states_130 = None
        hidden_states_ln_17 = torch.nn.functional.layer_norm(
            hidden_states_135,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_102 = torch._C._nn.linear(
            hidden_states_ln_17,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_68 = linear_102.view((1, -1, 20, 64))
        linear_102 = None
        query_layer_51 = view_68.transpose(1, 2)
        view_68 = None
        linear_103 = torch._C._nn.linear(
            hidden_states_ln_17,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_69 = linear_103.view((1, -1, 20, 64))
        linear_103 = None
        key_layer_34 = view_69.transpose(1, 2)
        view_69 = None
        linear_104 = torch._C._nn.linear(
            hidden_states_ln_17,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_17 = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_70 = linear_104.view((1, -1, 20, 64))
        linear_104 = None
        value_layer_17 = view_70.transpose(1, 2)
        view_70 = None
        query_layer_52 = query_layer_51 * 0.125
        query_layer_51 = None
        arange_17 = torch.arange(13, device=device(type="cuda", index=0))
        t_17 = arange_17.type_as(
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_17 = None
        freqs_17 = torch.outer(
            t_17,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_17 = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_51 = torch.cat((freqs_17, freqs_17), dim=-1)
        freqs_17 = None
        emb_17 = cat_51.to(device(type="cuda", index=0))
        cat_51 = None
        cos_51 = emb_17.cos()
        getitem_172 = cos_51[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_51 = None
        sin_51 = emb_17.sin()
        emb_17 = None
        getitem_173 = sin_51[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_51 = None
        cos_52 = getitem_172[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_52 = getitem_173[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_124 = query_layer_52 * cos_52
        cos_52 = None
        chunk_34 = query_layer_52.chunk(2, dim=-1)
        query_layer_52 = None
        x1_34 = chunk_34[0]
        x2_34 = chunk_34[1]
        chunk_34 = None
        neg_34 = -x2_34
        x2_34 = None
        cat_52 = torch.cat((neg_34, x1_34), dim=-1)
        neg_34 = x1_34 = None
        mul_125 = cat_52 * sin_52
        cat_52 = sin_52 = None
        add_103 = mul_124 + mul_125
        mul_124 = mul_125 = None
        query_layer_53 = add_103.to(dtype=torch.float32)
        add_103 = None
        cos_53 = getitem_172[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_53 = getitem_173[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_126 = key_layer_34 * cos_53
        cos_53 = None
        chunk_35 = key_layer_34.chunk(2, dim=-1)
        key_layer_34 = None
        x1_35 = chunk_35[0]
        x2_35 = chunk_35[1]
        chunk_35 = None
        neg_35 = -x2_35
        x2_35 = None
        cat_53 = torch.cat((neg_35, x1_35), dim=-1)
        neg_35 = x1_35 = None
        mul_127 = cat_53 * sin_53
        cat_53 = sin_53 = None
        add_104 = mul_126 + mul_127
        mul_126 = mul_127 = None
        key_layer_35 = add_104.to(dtype=torch.float32)
        add_104 = None
        transpose_71 = key_layer_35.transpose(-1, -2)
        key_layer_35 = None
        attention_scores_34 = torch.matmul(query_layer_53, transpose_71)
        query_layer_53 = transpose_71 = None
        attention_scores_35 = attention_scores_34 + extended_attention_mask_2
        attention_scores_34 = None
        attention_probs_34 = torch.nn.functional.softmax(attention_scores_35, dim=-1)
        attention_scores_35 = None
        attention_probs_35 = torch.nn.functional.dropout(
            attention_probs_34, 0.0, False, False
        )
        attention_probs_34 = None
        to_74 = attention_probs_35.to(torch.float32)
        attention_probs_35 = None
        context_layer_51 = torch.matmul(to_74, value_layer_17)
        to_74 = value_layer_17 = None
        permute_17 = context_layer_51.permute(0, 2, 1, 3)
        context_layer_51 = None
        context_layer_52 = permute_17.contiguous()
        permute_17 = None
        context_layer_53 = context_layer_52.view((1, 13, 1280))
        context_layer_52 = None
        hidden_states_136 = torch._C._nn.linear(
            context_layer_53,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_53 = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_137 = torch.nn.functional.dropout(
            hidden_states_136, 0.0, False, False
        )
        hidden_states_136 = None
        hidden_states_138 = hidden_states_137 + hidden_states_135
        hidden_states_137 = hidden_states_135 = None
        attention_output_ln_17 = torch.nn.functional.layer_norm(
            hidden_states_138,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_17_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_139 = torch._C._nn.linear(
            attention_output_ln_17,
            l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_17 = l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_128 = hidden_states_139 * 0.5
        truediv_19 = hidden_states_139 / 1.4142135623730951
        hidden_states_139 = None
        erf_17 = torch.erf(truediv_19)
        truediv_19 = None
        add_107 = 1.0 + erf_17
        erf_17 = None
        hidden_states_140 = mul_128 * add_107
        mul_128 = add_107 = None
        hidden_states_141 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_140 = l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_142 = torch.nn.functional.dropout(
            hidden_states_141, 0.0, False, False
        )
        hidden_states_141 = None
        hidden_states_143 = hidden_states_142 + hidden_states_138
        hidden_states_142 = hidden_states_138 = None
        hidden_states_ln_18 = torch.nn.functional.layer_norm(
            hidden_states_143,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_108 = torch._C._nn.linear(
            hidden_states_ln_18,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_72 = linear_108.view((1, -1, 20, 64))
        linear_108 = None
        query_layer_54 = view_72.transpose(1, 2)
        view_72 = None
        linear_109 = torch._C._nn.linear(
            hidden_states_ln_18,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_73 = linear_109.view((1, -1, 20, 64))
        linear_109 = None
        key_layer_36 = view_73.transpose(1, 2)
        view_73 = None
        linear_110 = torch._C._nn.linear(
            hidden_states_ln_18,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_18 = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_74 = linear_110.view((1, -1, 20, 64))
        linear_110 = None
        value_layer_18 = view_74.transpose(1, 2)
        view_74 = None
        query_layer_55 = query_layer_54 * 0.125
        query_layer_54 = None
        arange_18 = torch.arange(13, device=device(type="cuda", index=0))
        t_18 = arange_18.type_as(
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_18 = None
        freqs_18 = torch.outer(
            t_18,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_18 = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_54 = torch.cat((freqs_18, freqs_18), dim=-1)
        freqs_18 = None
        emb_18 = cat_54.to(device(type="cuda", index=0))
        cat_54 = None
        cos_54 = emb_18.cos()
        getitem_182 = cos_54[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_54 = None
        sin_54 = emb_18.sin()
        emb_18 = None
        getitem_183 = sin_54[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_54 = None
        cos_55 = getitem_182[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_55 = getitem_183[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_131 = query_layer_55 * cos_55
        cos_55 = None
        chunk_36 = query_layer_55.chunk(2, dim=-1)
        query_layer_55 = None
        x1_36 = chunk_36[0]
        x2_36 = chunk_36[1]
        chunk_36 = None
        neg_36 = -x2_36
        x2_36 = None
        cat_55 = torch.cat((neg_36, x1_36), dim=-1)
        neg_36 = x1_36 = None
        mul_132 = cat_55 * sin_55
        cat_55 = sin_55 = None
        add_109 = mul_131 + mul_132
        mul_131 = mul_132 = None
        query_layer_56 = add_109.to(dtype=torch.float32)
        add_109 = None
        cos_56 = getitem_182[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_56 = getitem_183[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_133 = key_layer_36 * cos_56
        cos_56 = None
        chunk_37 = key_layer_36.chunk(2, dim=-1)
        key_layer_36 = None
        x1_37 = chunk_37[0]
        x2_37 = chunk_37[1]
        chunk_37 = None
        neg_37 = -x2_37
        x2_37 = None
        cat_56 = torch.cat((neg_37, x1_37), dim=-1)
        neg_37 = x1_37 = None
        mul_134 = cat_56 * sin_56
        cat_56 = sin_56 = None
        add_110 = mul_133 + mul_134
        mul_133 = mul_134 = None
        key_layer_37 = add_110.to(dtype=torch.float32)
        add_110 = None
        transpose_75 = key_layer_37.transpose(-1, -2)
        key_layer_37 = None
        attention_scores_36 = torch.matmul(query_layer_56, transpose_75)
        query_layer_56 = transpose_75 = None
        attention_scores_37 = attention_scores_36 + extended_attention_mask_2
        attention_scores_36 = None
        attention_probs_36 = torch.nn.functional.softmax(attention_scores_37, dim=-1)
        attention_scores_37 = None
        attention_probs_37 = torch.nn.functional.dropout(
            attention_probs_36, 0.0, False, False
        )
        attention_probs_36 = None
        to_78 = attention_probs_37.to(torch.float32)
        attention_probs_37 = None
        context_layer_54 = torch.matmul(to_78, value_layer_18)
        to_78 = value_layer_18 = None
        permute_18 = context_layer_54.permute(0, 2, 1, 3)
        context_layer_54 = None
        context_layer_55 = permute_18.contiguous()
        permute_18 = None
        context_layer_56 = context_layer_55.view((1, 13, 1280))
        context_layer_55 = None
        hidden_states_144 = torch._C._nn.linear(
            context_layer_56,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_56 = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_145 = torch.nn.functional.dropout(
            hidden_states_144, 0.0, False, False
        )
        hidden_states_144 = None
        hidden_states_146 = hidden_states_145 + hidden_states_143
        hidden_states_145 = hidden_states_143 = None
        attention_output_ln_18 = torch.nn.functional.layer_norm(
            hidden_states_146,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_18_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_147 = torch._C._nn.linear(
            attention_output_ln_18,
            l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_18 = l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_135 = hidden_states_147 * 0.5
        truediv_20 = hidden_states_147 / 1.4142135623730951
        hidden_states_147 = None
        erf_18 = torch.erf(truediv_20)
        truediv_20 = None
        add_113 = 1.0 + erf_18
        erf_18 = None
        hidden_states_148 = mul_135 * add_113
        mul_135 = add_113 = None
        hidden_states_149 = torch._C._nn.linear(
            hidden_states_148,
            l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_148 = l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_150 = torch.nn.functional.dropout(
            hidden_states_149, 0.0, False, False
        )
        hidden_states_149 = None
        hidden_states_151 = hidden_states_150 + hidden_states_146
        hidden_states_150 = hidden_states_146 = None
        hidden_states_ln_19 = torch.nn.functional.layer_norm(
            hidden_states_151,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_114 = torch._C._nn.linear(
            hidden_states_ln_19,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_76 = linear_114.view((1, -1, 20, 64))
        linear_114 = None
        query_layer_57 = view_76.transpose(1, 2)
        view_76 = None
        linear_115 = torch._C._nn.linear(
            hidden_states_ln_19,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_77 = linear_115.view((1, -1, 20, 64))
        linear_115 = None
        key_layer_38 = view_77.transpose(1, 2)
        view_77 = None
        linear_116 = torch._C._nn.linear(
            hidden_states_ln_19,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_19 = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_78 = linear_116.view((1, -1, 20, 64))
        linear_116 = None
        value_layer_19 = view_78.transpose(1, 2)
        view_78 = None
        query_layer_58 = query_layer_57 * 0.125
        query_layer_57 = None
        arange_19 = torch.arange(13, device=device(type="cuda", index=0))
        t_19 = arange_19.type_as(
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_19 = None
        freqs_19 = torch.outer(
            t_19,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_19 = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_57 = torch.cat((freqs_19, freqs_19), dim=-1)
        freqs_19 = None
        emb_19 = cat_57.to(device(type="cuda", index=0))
        cat_57 = None
        cos_57 = emb_19.cos()
        getitem_192 = cos_57[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_57 = None
        sin_57 = emb_19.sin()
        emb_19 = None
        getitem_193 = sin_57[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_57 = None
        cos_58 = getitem_192[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_58 = getitem_193[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_138 = query_layer_58 * cos_58
        cos_58 = None
        chunk_38 = query_layer_58.chunk(2, dim=-1)
        query_layer_58 = None
        x1_38 = chunk_38[0]
        x2_38 = chunk_38[1]
        chunk_38 = None
        neg_38 = -x2_38
        x2_38 = None
        cat_58 = torch.cat((neg_38, x1_38), dim=-1)
        neg_38 = x1_38 = None
        mul_139 = cat_58 * sin_58
        cat_58 = sin_58 = None
        add_115 = mul_138 + mul_139
        mul_138 = mul_139 = None
        query_layer_59 = add_115.to(dtype=torch.float32)
        add_115 = None
        cos_59 = getitem_192[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_59 = getitem_193[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_140 = key_layer_38 * cos_59
        cos_59 = None
        chunk_39 = key_layer_38.chunk(2, dim=-1)
        key_layer_38 = None
        x1_39 = chunk_39[0]
        x2_39 = chunk_39[1]
        chunk_39 = None
        neg_39 = -x2_39
        x2_39 = None
        cat_59 = torch.cat((neg_39, x1_39), dim=-1)
        neg_39 = x1_39 = None
        mul_141 = cat_59 * sin_59
        cat_59 = sin_59 = None
        add_116 = mul_140 + mul_141
        mul_140 = mul_141 = None
        key_layer_39 = add_116.to(dtype=torch.float32)
        add_116 = None
        transpose_79 = key_layer_39.transpose(-1, -2)
        key_layer_39 = None
        attention_scores_38 = torch.matmul(query_layer_59, transpose_79)
        query_layer_59 = transpose_79 = None
        attention_scores_39 = attention_scores_38 + extended_attention_mask_2
        attention_scores_38 = None
        attention_probs_38 = torch.nn.functional.softmax(attention_scores_39, dim=-1)
        attention_scores_39 = None
        attention_probs_39 = torch.nn.functional.dropout(
            attention_probs_38, 0.0, False, False
        )
        attention_probs_38 = None
        to_82 = attention_probs_39.to(torch.float32)
        attention_probs_39 = None
        context_layer_57 = torch.matmul(to_82, value_layer_19)
        to_82 = value_layer_19 = None
        permute_19 = context_layer_57.permute(0, 2, 1, 3)
        context_layer_57 = None
        context_layer_58 = permute_19.contiguous()
        permute_19 = None
        context_layer_59 = context_layer_58.view((1, 13, 1280))
        context_layer_58 = None
        hidden_states_152 = torch._C._nn.linear(
            context_layer_59,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_59 = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_153 = torch.nn.functional.dropout(
            hidden_states_152, 0.0, False, False
        )
        hidden_states_152 = None
        hidden_states_154 = hidden_states_153 + hidden_states_151
        hidden_states_153 = hidden_states_151 = None
        attention_output_ln_19 = torch.nn.functional.layer_norm(
            hidden_states_154,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_19_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_155 = torch._C._nn.linear(
            attention_output_ln_19,
            l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_19 = l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_142 = hidden_states_155 * 0.5
        truediv_21 = hidden_states_155 / 1.4142135623730951
        hidden_states_155 = None
        erf_19 = torch.erf(truediv_21)
        truediv_21 = None
        add_119 = 1.0 + erf_19
        erf_19 = None
        hidden_states_156 = mul_142 * add_119
        mul_142 = add_119 = None
        hidden_states_157 = torch._C._nn.linear(
            hidden_states_156,
            l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_156 = l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_158 = torch.nn.functional.dropout(
            hidden_states_157, 0.0, False, False
        )
        hidden_states_157 = None
        hidden_states_159 = hidden_states_158 + hidden_states_154
        hidden_states_158 = hidden_states_154 = None
        hidden_states_ln_20 = torch.nn.functional.layer_norm(
            hidden_states_159,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_120 = torch._C._nn.linear(
            hidden_states_ln_20,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_80 = linear_120.view((1, -1, 20, 64))
        linear_120 = None
        query_layer_60 = view_80.transpose(1, 2)
        view_80 = None
        linear_121 = torch._C._nn.linear(
            hidden_states_ln_20,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_81 = linear_121.view((1, -1, 20, 64))
        linear_121 = None
        key_layer_40 = view_81.transpose(1, 2)
        view_81 = None
        linear_122 = torch._C._nn.linear(
            hidden_states_ln_20,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_20 = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_82 = linear_122.view((1, -1, 20, 64))
        linear_122 = None
        value_layer_20 = view_82.transpose(1, 2)
        view_82 = None
        query_layer_61 = query_layer_60 * 0.125
        query_layer_60 = None
        arange_20 = torch.arange(13, device=device(type="cuda", index=0))
        t_20 = arange_20.type_as(
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_20 = None
        freqs_20 = torch.outer(
            t_20,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_20 = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_60 = torch.cat((freqs_20, freqs_20), dim=-1)
        freqs_20 = None
        emb_20 = cat_60.to(device(type="cuda", index=0))
        cat_60 = None
        cos_60 = emb_20.cos()
        getitem_202 = cos_60[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_60 = None
        sin_60 = emb_20.sin()
        emb_20 = None
        getitem_203 = sin_60[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_60 = None
        cos_61 = getitem_202[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_61 = getitem_203[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_145 = query_layer_61 * cos_61
        cos_61 = None
        chunk_40 = query_layer_61.chunk(2, dim=-1)
        query_layer_61 = None
        x1_40 = chunk_40[0]
        x2_40 = chunk_40[1]
        chunk_40 = None
        neg_40 = -x2_40
        x2_40 = None
        cat_61 = torch.cat((neg_40, x1_40), dim=-1)
        neg_40 = x1_40 = None
        mul_146 = cat_61 * sin_61
        cat_61 = sin_61 = None
        add_121 = mul_145 + mul_146
        mul_145 = mul_146 = None
        query_layer_62 = add_121.to(dtype=torch.float32)
        add_121 = None
        cos_62 = getitem_202[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_62 = getitem_203[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_147 = key_layer_40 * cos_62
        cos_62 = None
        chunk_41 = key_layer_40.chunk(2, dim=-1)
        key_layer_40 = None
        x1_41 = chunk_41[0]
        x2_41 = chunk_41[1]
        chunk_41 = None
        neg_41 = -x2_41
        x2_41 = None
        cat_62 = torch.cat((neg_41, x1_41), dim=-1)
        neg_41 = x1_41 = None
        mul_148 = cat_62 * sin_62
        cat_62 = sin_62 = None
        add_122 = mul_147 + mul_148
        mul_147 = mul_148 = None
        key_layer_41 = add_122.to(dtype=torch.float32)
        add_122 = None
        transpose_83 = key_layer_41.transpose(-1, -2)
        key_layer_41 = None
        attention_scores_40 = torch.matmul(query_layer_62, transpose_83)
        query_layer_62 = transpose_83 = None
        attention_scores_41 = attention_scores_40 + extended_attention_mask_2
        attention_scores_40 = None
        attention_probs_40 = torch.nn.functional.softmax(attention_scores_41, dim=-1)
        attention_scores_41 = None
        attention_probs_41 = torch.nn.functional.dropout(
            attention_probs_40, 0.0, False, False
        )
        attention_probs_40 = None
        to_86 = attention_probs_41.to(torch.float32)
        attention_probs_41 = None
        context_layer_60 = torch.matmul(to_86, value_layer_20)
        to_86 = value_layer_20 = None
        permute_20 = context_layer_60.permute(0, 2, 1, 3)
        context_layer_60 = None
        context_layer_61 = permute_20.contiguous()
        permute_20 = None
        context_layer_62 = context_layer_61.view((1, 13, 1280))
        context_layer_61 = None
        hidden_states_160 = torch._C._nn.linear(
            context_layer_62,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_62 = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_161 = torch.nn.functional.dropout(
            hidden_states_160, 0.0, False, False
        )
        hidden_states_160 = None
        hidden_states_162 = hidden_states_161 + hidden_states_159
        hidden_states_161 = hidden_states_159 = None
        attention_output_ln_20 = torch.nn.functional.layer_norm(
            hidden_states_162,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_20_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_163 = torch._C._nn.linear(
            attention_output_ln_20,
            l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_20 = l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_149 = hidden_states_163 * 0.5
        truediv_22 = hidden_states_163 / 1.4142135623730951
        hidden_states_163 = None
        erf_20 = torch.erf(truediv_22)
        truediv_22 = None
        add_125 = 1.0 + erf_20
        erf_20 = None
        hidden_states_164 = mul_149 * add_125
        mul_149 = add_125 = None
        hidden_states_165 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_164 = l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_166 = torch.nn.functional.dropout(
            hidden_states_165, 0.0, False, False
        )
        hidden_states_165 = None
        hidden_states_167 = hidden_states_166 + hidden_states_162
        hidden_states_166 = hidden_states_162 = None
        hidden_states_ln_21 = torch.nn.functional.layer_norm(
            hidden_states_167,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_126 = torch._C._nn.linear(
            hidden_states_ln_21,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_84 = linear_126.view((1, -1, 20, 64))
        linear_126 = None
        query_layer_63 = view_84.transpose(1, 2)
        view_84 = None
        linear_127 = torch._C._nn.linear(
            hidden_states_ln_21,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_85 = linear_127.view((1, -1, 20, 64))
        linear_127 = None
        key_layer_42 = view_85.transpose(1, 2)
        view_85 = None
        linear_128 = torch._C._nn.linear(
            hidden_states_ln_21,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_21 = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_86 = linear_128.view((1, -1, 20, 64))
        linear_128 = None
        value_layer_21 = view_86.transpose(1, 2)
        view_86 = None
        query_layer_64 = query_layer_63 * 0.125
        query_layer_63 = None
        arange_21 = torch.arange(13, device=device(type="cuda", index=0))
        t_21 = arange_21.type_as(
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_21 = None
        freqs_21 = torch.outer(
            t_21,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_21 = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_63 = torch.cat((freqs_21, freqs_21), dim=-1)
        freqs_21 = None
        emb_21 = cat_63.to(device(type="cuda", index=0))
        cat_63 = None
        cos_63 = emb_21.cos()
        getitem_212 = cos_63[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_63 = None
        sin_63 = emb_21.sin()
        emb_21 = None
        getitem_213 = sin_63[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_63 = None
        cos_64 = getitem_212[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_64 = getitem_213[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_152 = query_layer_64 * cos_64
        cos_64 = None
        chunk_42 = query_layer_64.chunk(2, dim=-1)
        query_layer_64 = None
        x1_42 = chunk_42[0]
        x2_42 = chunk_42[1]
        chunk_42 = None
        neg_42 = -x2_42
        x2_42 = None
        cat_64 = torch.cat((neg_42, x1_42), dim=-1)
        neg_42 = x1_42 = None
        mul_153 = cat_64 * sin_64
        cat_64 = sin_64 = None
        add_127 = mul_152 + mul_153
        mul_152 = mul_153 = None
        query_layer_65 = add_127.to(dtype=torch.float32)
        add_127 = None
        cos_65 = getitem_212[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_65 = getitem_213[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_154 = key_layer_42 * cos_65
        cos_65 = None
        chunk_43 = key_layer_42.chunk(2, dim=-1)
        key_layer_42 = None
        x1_43 = chunk_43[0]
        x2_43 = chunk_43[1]
        chunk_43 = None
        neg_43 = -x2_43
        x2_43 = None
        cat_65 = torch.cat((neg_43, x1_43), dim=-1)
        neg_43 = x1_43 = None
        mul_155 = cat_65 * sin_65
        cat_65 = sin_65 = None
        add_128 = mul_154 + mul_155
        mul_154 = mul_155 = None
        key_layer_43 = add_128.to(dtype=torch.float32)
        add_128 = None
        transpose_87 = key_layer_43.transpose(-1, -2)
        key_layer_43 = None
        attention_scores_42 = torch.matmul(query_layer_65, transpose_87)
        query_layer_65 = transpose_87 = None
        attention_scores_43 = attention_scores_42 + extended_attention_mask_2
        attention_scores_42 = None
        attention_probs_42 = torch.nn.functional.softmax(attention_scores_43, dim=-1)
        attention_scores_43 = None
        attention_probs_43 = torch.nn.functional.dropout(
            attention_probs_42, 0.0, False, False
        )
        attention_probs_42 = None
        to_90 = attention_probs_43.to(torch.float32)
        attention_probs_43 = None
        context_layer_63 = torch.matmul(to_90, value_layer_21)
        to_90 = value_layer_21 = None
        permute_21 = context_layer_63.permute(0, 2, 1, 3)
        context_layer_63 = None
        context_layer_64 = permute_21.contiguous()
        permute_21 = None
        context_layer_65 = context_layer_64.view((1, 13, 1280))
        context_layer_64 = None
        hidden_states_168 = torch._C._nn.linear(
            context_layer_65,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_65 = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_169 = torch.nn.functional.dropout(
            hidden_states_168, 0.0, False, False
        )
        hidden_states_168 = None
        hidden_states_170 = hidden_states_169 + hidden_states_167
        hidden_states_169 = hidden_states_167 = None
        attention_output_ln_21 = torch.nn.functional.layer_norm(
            hidden_states_170,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_21_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_171 = torch._C._nn.linear(
            attention_output_ln_21,
            l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_21 = l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_156 = hidden_states_171 * 0.5
        truediv_23 = hidden_states_171 / 1.4142135623730951
        hidden_states_171 = None
        erf_21 = torch.erf(truediv_23)
        truediv_23 = None
        add_131 = 1.0 + erf_21
        erf_21 = None
        hidden_states_172 = mul_156 * add_131
        mul_156 = add_131 = None
        hidden_states_173 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_172 = l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_174 = torch.nn.functional.dropout(
            hidden_states_173, 0.0, False, False
        )
        hidden_states_173 = None
        hidden_states_175 = hidden_states_174 + hidden_states_170
        hidden_states_174 = hidden_states_170 = None
        hidden_states_ln_22 = torch.nn.functional.layer_norm(
            hidden_states_175,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_132 = torch._C._nn.linear(
            hidden_states_ln_22,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_88 = linear_132.view((1, -1, 20, 64))
        linear_132 = None
        query_layer_66 = view_88.transpose(1, 2)
        view_88 = None
        linear_133 = torch._C._nn.linear(
            hidden_states_ln_22,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_89 = linear_133.view((1, -1, 20, 64))
        linear_133 = None
        key_layer_44 = view_89.transpose(1, 2)
        view_89 = None
        linear_134 = torch._C._nn.linear(
            hidden_states_ln_22,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_22 = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_90 = linear_134.view((1, -1, 20, 64))
        linear_134 = None
        value_layer_22 = view_90.transpose(1, 2)
        view_90 = None
        query_layer_67 = query_layer_66 * 0.125
        query_layer_66 = None
        arange_22 = torch.arange(13, device=device(type="cuda", index=0))
        t_22 = arange_22.type_as(
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_22 = None
        freqs_22 = torch.outer(
            t_22,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_22 = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_66 = torch.cat((freqs_22, freqs_22), dim=-1)
        freqs_22 = None
        emb_22 = cat_66.to(device(type="cuda", index=0))
        cat_66 = None
        cos_66 = emb_22.cos()
        getitem_222 = cos_66[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_66 = None
        sin_66 = emb_22.sin()
        emb_22 = None
        getitem_223 = sin_66[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_66 = None
        cos_67 = getitem_222[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_67 = getitem_223[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_159 = query_layer_67 * cos_67
        cos_67 = None
        chunk_44 = query_layer_67.chunk(2, dim=-1)
        query_layer_67 = None
        x1_44 = chunk_44[0]
        x2_44 = chunk_44[1]
        chunk_44 = None
        neg_44 = -x2_44
        x2_44 = None
        cat_67 = torch.cat((neg_44, x1_44), dim=-1)
        neg_44 = x1_44 = None
        mul_160 = cat_67 * sin_67
        cat_67 = sin_67 = None
        add_133 = mul_159 + mul_160
        mul_159 = mul_160 = None
        query_layer_68 = add_133.to(dtype=torch.float32)
        add_133 = None
        cos_68 = getitem_222[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_68 = getitem_223[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_161 = key_layer_44 * cos_68
        cos_68 = None
        chunk_45 = key_layer_44.chunk(2, dim=-1)
        key_layer_44 = None
        x1_45 = chunk_45[0]
        x2_45 = chunk_45[1]
        chunk_45 = None
        neg_45 = -x2_45
        x2_45 = None
        cat_68 = torch.cat((neg_45, x1_45), dim=-1)
        neg_45 = x1_45 = None
        mul_162 = cat_68 * sin_68
        cat_68 = sin_68 = None
        add_134 = mul_161 + mul_162
        mul_161 = mul_162 = None
        key_layer_45 = add_134.to(dtype=torch.float32)
        add_134 = None
        transpose_91 = key_layer_45.transpose(-1, -2)
        key_layer_45 = None
        attention_scores_44 = torch.matmul(query_layer_68, transpose_91)
        query_layer_68 = transpose_91 = None
        attention_scores_45 = attention_scores_44 + extended_attention_mask_2
        attention_scores_44 = None
        attention_probs_44 = torch.nn.functional.softmax(attention_scores_45, dim=-1)
        attention_scores_45 = None
        attention_probs_45 = torch.nn.functional.dropout(
            attention_probs_44, 0.0, False, False
        )
        attention_probs_44 = None
        to_94 = attention_probs_45.to(torch.float32)
        attention_probs_45 = None
        context_layer_66 = torch.matmul(to_94, value_layer_22)
        to_94 = value_layer_22 = None
        permute_22 = context_layer_66.permute(0, 2, 1, 3)
        context_layer_66 = None
        context_layer_67 = permute_22.contiguous()
        permute_22 = None
        context_layer_68 = context_layer_67.view((1, 13, 1280))
        context_layer_67 = None
        hidden_states_176 = torch._C._nn.linear(
            context_layer_68,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_68 = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_177 = torch.nn.functional.dropout(
            hidden_states_176, 0.0, False, False
        )
        hidden_states_176 = None
        hidden_states_178 = hidden_states_177 + hidden_states_175
        hidden_states_177 = hidden_states_175 = None
        attention_output_ln_22 = torch.nn.functional.layer_norm(
            hidden_states_178,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_22_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_179 = torch._C._nn.linear(
            attention_output_ln_22,
            l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_22 = l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_163 = hidden_states_179 * 0.5
        truediv_24 = hidden_states_179 / 1.4142135623730951
        hidden_states_179 = None
        erf_22 = torch.erf(truediv_24)
        truediv_24 = None
        add_137 = 1.0 + erf_22
        erf_22 = None
        hidden_states_180 = mul_163 * add_137
        mul_163 = add_137 = None
        hidden_states_181 = torch._C._nn.linear(
            hidden_states_180,
            l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_180 = l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_182 = torch.nn.functional.dropout(
            hidden_states_181, 0.0, False, False
        )
        hidden_states_181 = None
        hidden_states_183 = hidden_states_182 + hidden_states_178
        hidden_states_182 = hidden_states_178 = None
        hidden_states_ln_23 = torch.nn.functional.layer_norm(
            hidden_states_183,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_138 = torch._C._nn.linear(
            hidden_states_ln_23,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_92 = linear_138.view((1, -1, 20, 64))
        linear_138 = None
        query_layer_69 = view_92.transpose(1, 2)
        view_92 = None
        linear_139 = torch._C._nn.linear(
            hidden_states_ln_23,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_93 = linear_139.view((1, -1, 20, 64))
        linear_139 = None
        key_layer_46 = view_93.transpose(1, 2)
        view_93 = None
        linear_140 = torch._C._nn.linear(
            hidden_states_ln_23,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_23 = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_94 = linear_140.view((1, -1, 20, 64))
        linear_140 = None
        value_layer_23 = view_94.transpose(1, 2)
        view_94 = None
        query_layer_70 = query_layer_69 * 0.125
        query_layer_69 = None
        arange_23 = torch.arange(13, device=device(type="cuda", index=0))
        t_23 = arange_23.type_as(
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_23 = None
        freqs_23 = torch.outer(
            t_23,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_23 = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_69 = torch.cat((freqs_23, freqs_23), dim=-1)
        freqs_23 = None
        emb_23 = cat_69.to(device(type="cuda", index=0))
        cat_69 = None
        cos_69 = emb_23.cos()
        getitem_232 = cos_69[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_69 = None
        sin_69 = emb_23.sin()
        emb_23 = None
        getitem_233 = sin_69[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_69 = None
        cos_70 = getitem_232[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_70 = getitem_233[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_166 = query_layer_70 * cos_70
        cos_70 = None
        chunk_46 = query_layer_70.chunk(2, dim=-1)
        query_layer_70 = None
        x1_46 = chunk_46[0]
        x2_46 = chunk_46[1]
        chunk_46 = None
        neg_46 = -x2_46
        x2_46 = None
        cat_70 = torch.cat((neg_46, x1_46), dim=-1)
        neg_46 = x1_46 = None
        mul_167 = cat_70 * sin_70
        cat_70 = sin_70 = None
        add_139 = mul_166 + mul_167
        mul_166 = mul_167 = None
        query_layer_71 = add_139.to(dtype=torch.float32)
        add_139 = None
        cos_71 = getitem_232[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_71 = getitem_233[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_168 = key_layer_46 * cos_71
        cos_71 = None
        chunk_47 = key_layer_46.chunk(2, dim=-1)
        key_layer_46 = None
        x1_47 = chunk_47[0]
        x2_47 = chunk_47[1]
        chunk_47 = None
        neg_47 = -x2_47
        x2_47 = None
        cat_71 = torch.cat((neg_47, x1_47), dim=-1)
        neg_47 = x1_47 = None
        mul_169 = cat_71 * sin_71
        cat_71 = sin_71 = None
        add_140 = mul_168 + mul_169
        mul_168 = mul_169 = None
        key_layer_47 = add_140.to(dtype=torch.float32)
        add_140 = None
        transpose_95 = key_layer_47.transpose(-1, -2)
        key_layer_47 = None
        attention_scores_46 = torch.matmul(query_layer_71, transpose_95)
        query_layer_71 = transpose_95 = None
        attention_scores_47 = attention_scores_46 + extended_attention_mask_2
        attention_scores_46 = None
        attention_probs_46 = torch.nn.functional.softmax(attention_scores_47, dim=-1)
        attention_scores_47 = None
        attention_probs_47 = torch.nn.functional.dropout(
            attention_probs_46, 0.0, False, False
        )
        attention_probs_46 = None
        to_98 = attention_probs_47.to(torch.float32)
        attention_probs_47 = None
        context_layer_69 = torch.matmul(to_98, value_layer_23)
        to_98 = value_layer_23 = None
        permute_23 = context_layer_69.permute(0, 2, 1, 3)
        context_layer_69 = None
        context_layer_70 = permute_23.contiguous()
        permute_23 = None
        context_layer_71 = context_layer_70.view((1, 13, 1280))
        context_layer_70 = None
        hidden_states_184 = torch._C._nn.linear(
            context_layer_71,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_71 = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_185 = torch.nn.functional.dropout(
            hidden_states_184, 0.0, False, False
        )
        hidden_states_184 = None
        hidden_states_186 = hidden_states_185 + hidden_states_183
        hidden_states_185 = hidden_states_183 = None
        attention_output_ln_23 = torch.nn.functional.layer_norm(
            hidden_states_186,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_23_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_187 = torch._C._nn.linear(
            attention_output_ln_23,
            l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_23 = l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_170 = hidden_states_187 * 0.5
        truediv_25 = hidden_states_187 / 1.4142135623730951
        hidden_states_187 = None
        erf_23 = torch.erf(truediv_25)
        truediv_25 = None
        add_143 = 1.0 + erf_23
        erf_23 = None
        hidden_states_188 = mul_170 * add_143
        mul_170 = add_143 = None
        hidden_states_189 = torch._C._nn.linear(
            hidden_states_188,
            l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_188 = l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_190 = torch.nn.functional.dropout(
            hidden_states_189, 0.0, False, False
        )
        hidden_states_189 = None
        hidden_states_191 = hidden_states_190 + hidden_states_186
        hidden_states_190 = hidden_states_186 = None
        hidden_states_ln_24 = torch.nn.functional.layer_norm(
            hidden_states_191,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_144 = torch._C._nn.linear(
            hidden_states_ln_24,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_96 = linear_144.view((1, -1, 20, 64))
        linear_144 = None
        query_layer_72 = view_96.transpose(1, 2)
        view_96 = None
        linear_145 = torch._C._nn.linear(
            hidden_states_ln_24,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_97 = linear_145.view((1, -1, 20, 64))
        linear_145 = None
        key_layer_48 = view_97.transpose(1, 2)
        view_97 = None
        linear_146 = torch._C._nn.linear(
            hidden_states_ln_24,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_24 = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_98 = linear_146.view((1, -1, 20, 64))
        linear_146 = None
        value_layer_24 = view_98.transpose(1, 2)
        view_98 = None
        query_layer_73 = query_layer_72 * 0.125
        query_layer_72 = None
        arange_24 = torch.arange(13, device=device(type="cuda", index=0))
        t_24 = arange_24.type_as(
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_24 = None
        freqs_24 = torch.outer(
            t_24,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_24 = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_72 = torch.cat((freqs_24, freqs_24), dim=-1)
        freqs_24 = None
        emb_24 = cat_72.to(device(type="cuda", index=0))
        cat_72 = None
        cos_72 = emb_24.cos()
        getitem_242 = cos_72[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_72 = None
        sin_72 = emb_24.sin()
        emb_24 = None
        getitem_243 = sin_72[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_72 = None
        cos_73 = getitem_242[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_73 = getitem_243[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_173 = query_layer_73 * cos_73
        cos_73 = None
        chunk_48 = query_layer_73.chunk(2, dim=-1)
        query_layer_73 = None
        x1_48 = chunk_48[0]
        x2_48 = chunk_48[1]
        chunk_48 = None
        neg_48 = -x2_48
        x2_48 = None
        cat_73 = torch.cat((neg_48, x1_48), dim=-1)
        neg_48 = x1_48 = None
        mul_174 = cat_73 * sin_73
        cat_73 = sin_73 = None
        add_145 = mul_173 + mul_174
        mul_173 = mul_174 = None
        query_layer_74 = add_145.to(dtype=torch.float32)
        add_145 = None
        cos_74 = getitem_242[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_74 = getitem_243[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_175 = key_layer_48 * cos_74
        cos_74 = None
        chunk_49 = key_layer_48.chunk(2, dim=-1)
        key_layer_48 = None
        x1_49 = chunk_49[0]
        x2_49 = chunk_49[1]
        chunk_49 = None
        neg_49 = -x2_49
        x2_49 = None
        cat_74 = torch.cat((neg_49, x1_49), dim=-1)
        neg_49 = x1_49 = None
        mul_176 = cat_74 * sin_74
        cat_74 = sin_74 = None
        add_146 = mul_175 + mul_176
        mul_175 = mul_176 = None
        key_layer_49 = add_146.to(dtype=torch.float32)
        add_146 = None
        transpose_99 = key_layer_49.transpose(-1, -2)
        key_layer_49 = None
        attention_scores_48 = torch.matmul(query_layer_74, transpose_99)
        query_layer_74 = transpose_99 = None
        attention_scores_49 = attention_scores_48 + extended_attention_mask_2
        attention_scores_48 = None
        attention_probs_48 = torch.nn.functional.softmax(attention_scores_49, dim=-1)
        attention_scores_49 = None
        attention_probs_49 = torch.nn.functional.dropout(
            attention_probs_48, 0.0, False, False
        )
        attention_probs_48 = None
        to_102 = attention_probs_49.to(torch.float32)
        attention_probs_49 = None
        context_layer_72 = torch.matmul(to_102, value_layer_24)
        to_102 = value_layer_24 = None
        permute_24 = context_layer_72.permute(0, 2, 1, 3)
        context_layer_72 = None
        context_layer_73 = permute_24.contiguous()
        permute_24 = None
        context_layer_74 = context_layer_73.view((1, 13, 1280))
        context_layer_73 = None
        hidden_states_192 = torch._C._nn.linear(
            context_layer_74,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_74 = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_193 = torch.nn.functional.dropout(
            hidden_states_192, 0.0, False, False
        )
        hidden_states_192 = None
        hidden_states_194 = hidden_states_193 + hidden_states_191
        hidden_states_193 = hidden_states_191 = None
        attention_output_ln_24 = torch.nn.functional.layer_norm(
            hidden_states_194,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_24_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_195 = torch._C._nn.linear(
            attention_output_ln_24,
            l_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_24 = l_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_177 = hidden_states_195 * 0.5
        truediv_26 = hidden_states_195 / 1.4142135623730951
        hidden_states_195 = None
        erf_24 = torch.erf(truediv_26)
        truediv_26 = None
        add_149 = 1.0 + erf_24
        erf_24 = None
        hidden_states_196 = mul_177 * add_149
        mul_177 = add_149 = None
        hidden_states_197 = torch._C._nn.linear(
            hidden_states_196,
            l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_196 = l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_198 = torch.nn.functional.dropout(
            hidden_states_197, 0.0, False, False
        )
        hidden_states_197 = None
        hidden_states_199 = hidden_states_198 + hidden_states_194
        hidden_states_198 = hidden_states_194 = None
        hidden_states_ln_25 = torch.nn.functional.layer_norm(
            hidden_states_199,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_150 = torch._C._nn.linear(
            hidden_states_ln_25,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_100 = linear_150.view((1, -1, 20, 64))
        linear_150 = None
        query_layer_75 = view_100.transpose(1, 2)
        view_100 = None
        linear_151 = torch._C._nn.linear(
            hidden_states_ln_25,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_101 = linear_151.view((1, -1, 20, 64))
        linear_151 = None
        key_layer_50 = view_101.transpose(1, 2)
        view_101 = None
        linear_152 = torch._C._nn.linear(
            hidden_states_ln_25,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_25 = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_102 = linear_152.view((1, -1, 20, 64))
        linear_152 = None
        value_layer_25 = view_102.transpose(1, 2)
        view_102 = None
        query_layer_76 = query_layer_75 * 0.125
        query_layer_75 = None
        arange_25 = torch.arange(13, device=device(type="cuda", index=0))
        t_25 = arange_25.type_as(
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_25 = None
        freqs_25 = torch.outer(
            t_25,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_25 = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_75 = torch.cat((freqs_25, freqs_25), dim=-1)
        freqs_25 = None
        emb_25 = cat_75.to(device(type="cuda", index=0))
        cat_75 = None
        cos_75 = emb_25.cos()
        getitem_252 = cos_75[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_75 = None
        sin_75 = emb_25.sin()
        emb_25 = None
        getitem_253 = sin_75[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_75 = None
        cos_76 = getitem_252[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_76 = getitem_253[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_180 = query_layer_76 * cos_76
        cos_76 = None
        chunk_50 = query_layer_76.chunk(2, dim=-1)
        query_layer_76 = None
        x1_50 = chunk_50[0]
        x2_50 = chunk_50[1]
        chunk_50 = None
        neg_50 = -x2_50
        x2_50 = None
        cat_76 = torch.cat((neg_50, x1_50), dim=-1)
        neg_50 = x1_50 = None
        mul_181 = cat_76 * sin_76
        cat_76 = sin_76 = None
        add_151 = mul_180 + mul_181
        mul_180 = mul_181 = None
        query_layer_77 = add_151.to(dtype=torch.float32)
        add_151 = None
        cos_77 = getitem_252[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_77 = getitem_253[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_182 = key_layer_50 * cos_77
        cos_77 = None
        chunk_51 = key_layer_50.chunk(2, dim=-1)
        key_layer_50 = None
        x1_51 = chunk_51[0]
        x2_51 = chunk_51[1]
        chunk_51 = None
        neg_51 = -x2_51
        x2_51 = None
        cat_77 = torch.cat((neg_51, x1_51), dim=-1)
        neg_51 = x1_51 = None
        mul_183 = cat_77 * sin_77
        cat_77 = sin_77 = None
        add_152 = mul_182 + mul_183
        mul_182 = mul_183 = None
        key_layer_51 = add_152.to(dtype=torch.float32)
        add_152 = None
        transpose_103 = key_layer_51.transpose(-1, -2)
        key_layer_51 = None
        attention_scores_50 = torch.matmul(query_layer_77, transpose_103)
        query_layer_77 = transpose_103 = None
        attention_scores_51 = attention_scores_50 + extended_attention_mask_2
        attention_scores_50 = None
        attention_probs_50 = torch.nn.functional.softmax(attention_scores_51, dim=-1)
        attention_scores_51 = None
        attention_probs_51 = torch.nn.functional.dropout(
            attention_probs_50, 0.0, False, False
        )
        attention_probs_50 = None
        to_106 = attention_probs_51.to(torch.float32)
        attention_probs_51 = None
        context_layer_75 = torch.matmul(to_106, value_layer_25)
        to_106 = value_layer_25 = None
        permute_25 = context_layer_75.permute(0, 2, 1, 3)
        context_layer_75 = None
        context_layer_76 = permute_25.contiguous()
        permute_25 = None
        context_layer_77 = context_layer_76.view((1, 13, 1280))
        context_layer_76 = None
        hidden_states_200 = torch._C._nn.linear(
            context_layer_77,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_77 = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_201 = torch.nn.functional.dropout(
            hidden_states_200, 0.0, False, False
        )
        hidden_states_200 = None
        hidden_states_202 = hidden_states_201 + hidden_states_199
        hidden_states_201 = hidden_states_199 = None
        attention_output_ln_25 = torch.nn.functional.layer_norm(
            hidden_states_202,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_25_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_203 = torch._C._nn.linear(
            attention_output_ln_25,
            l_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_25 = l_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_184 = hidden_states_203 * 0.5
        truediv_27 = hidden_states_203 / 1.4142135623730951
        hidden_states_203 = None
        erf_25 = torch.erf(truediv_27)
        truediv_27 = None
        add_155 = 1.0 + erf_25
        erf_25 = None
        hidden_states_204 = mul_184 * add_155
        mul_184 = add_155 = None
        hidden_states_205 = torch._C._nn.linear(
            hidden_states_204,
            l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_204 = l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_206 = torch.nn.functional.dropout(
            hidden_states_205, 0.0, False, False
        )
        hidden_states_205 = None
        hidden_states_207 = hidden_states_206 + hidden_states_202
        hidden_states_206 = hidden_states_202 = None
        hidden_states_ln_26 = torch.nn.functional.layer_norm(
            hidden_states_207,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_156 = torch._C._nn.linear(
            hidden_states_ln_26,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_104 = linear_156.view((1, -1, 20, 64))
        linear_156 = None
        query_layer_78 = view_104.transpose(1, 2)
        view_104 = None
        linear_157 = torch._C._nn.linear(
            hidden_states_ln_26,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_105 = linear_157.view((1, -1, 20, 64))
        linear_157 = None
        key_layer_52 = view_105.transpose(1, 2)
        view_105 = None
        linear_158 = torch._C._nn.linear(
            hidden_states_ln_26,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_26 = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_106 = linear_158.view((1, -1, 20, 64))
        linear_158 = None
        value_layer_26 = view_106.transpose(1, 2)
        view_106 = None
        query_layer_79 = query_layer_78 * 0.125
        query_layer_78 = None
        arange_26 = torch.arange(13, device=device(type="cuda", index=0))
        t_26 = arange_26.type_as(
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_26 = None
        freqs_26 = torch.outer(
            t_26,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_26 = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_78 = torch.cat((freqs_26, freqs_26), dim=-1)
        freqs_26 = None
        emb_26 = cat_78.to(device(type="cuda", index=0))
        cat_78 = None
        cos_78 = emb_26.cos()
        getitem_262 = cos_78[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_78 = None
        sin_78 = emb_26.sin()
        emb_26 = None
        getitem_263 = sin_78[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_78 = None
        cos_79 = getitem_262[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_79 = getitem_263[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_187 = query_layer_79 * cos_79
        cos_79 = None
        chunk_52 = query_layer_79.chunk(2, dim=-1)
        query_layer_79 = None
        x1_52 = chunk_52[0]
        x2_52 = chunk_52[1]
        chunk_52 = None
        neg_52 = -x2_52
        x2_52 = None
        cat_79 = torch.cat((neg_52, x1_52), dim=-1)
        neg_52 = x1_52 = None
        mul_188 = cat_79 * sin_79
        cat_79 = sin_79 = None
        add_157 = mul_187 + mul_188
        mul_187 = mul_188 = None
        query_layer_80 = add_157.to(dtype=torch.float32)
        add_157 = None
        cos_80 = getitem_262[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_80 = getitem_263[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_189 = key_layer_52 * cos_80
        cos_80 = None
        chunk_53 = key_layer_52.chunk(2, dim=-1)
        key_layer_52 = None
        x1_53 = chunk_53[0]
        x2_53 = chunk_53[1]
        chunk_53 = None
        neg_53 = -x2_53
        x2_53 = None
        cat_80 = torch.cat((neg_53, x1_53), dim=-1)
        neg_53 = x1_53 = None
        mul_190 = cat_80 * sin_80
        cat_80 = sin_80 = None
        add_158 = mul_189 + mul_190
        mul_189 = mul_190 = None
        key_layer_53 = add_158.to(dtype=torch.float32)
        add_158 = None
        transpose_107 = key_layer_53.transpose(-1, -2)
        key_layer_53 = None
        attention_scores_52 = torch.matmul(query_layer_80, transpose_107)
        query_layer_80 = transpose_107 = None
        attention_scores_53 = attention_scores_52 + extended_attention_mask_2
        attention_scores_52 = None
        attention_probs_52 = torch.nn.functional.softmax(attention_scores_53, dim=-1)
        attention_scores_53 = None
        attention_probs_53 = torch.nn.functional.dropout(
            attention_probs_52, 0.0, False, False
        )
        attention_probs_52 = None
        to_110 = attention_probs_53.to(torch.float32)
        attention_probs_53 = None
        context_layer_78 = torch.matmul(to_110, value_layer_26)
        to_110 = value_layer_26 = None
        permute_26 = context_layer_78.permute(0, 2, 1, 3)
        context_layer_78 = None
        context_layer_79 = permute_26.contiguous()
        permute_26 = None
        context_layer_80 = context_layer_79.view((1, 13, 1280))
        context_layer_79 = None
        hidden_states_208 = torch._C._nn.linear(
            context_layer_80,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_80 = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_209 = torch.nn.functional.dropout(
            hidden_states_208, 0.0, False, False
        )
        hidden_states_208 = None
        hidden_states_210 = hidden_states_209 + hidden_states_207
        hidden_states_209 = hidden_states_207 = None
        attention_output_ln_26 = torch.nn.functional.layer_norm(
            hidden_states_210,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_26_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_211 = torch._C._nn.linear(
            attention_output_ln_26,
            l_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_26 = l_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_191 = hidden_states_211 * 0.5
        truediv_28 = hidden_states_211 / 1.4142135623730951
        hidden_states_211 = None
        erf_26 = torch.erf(truediv_28)
        truediv_28 = None
        add_161 = 1.0 + erf_26
        erf_26 = None
        hidden_states_212 = mul_191 * add_161
        mul_191 = add_161 = None
        hidden_states_213 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_212 = l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_214 = torch.nn.functional.dropout(
            hidden_states_213, 0.0, False, False
        )
        hidden_states_213 = None
        hidden_states_215 = hidden_states_214 + hidden_states_210
        hidden_states_214 = hidden_states_210 = None
        hidden_states_ln_27 = torch.nn.functional.layer_norm(
            hidden_states_215,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_162 = torch._C._nn.linear(
            hidden_states_ln_27,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_108 = linear_162.view((1, -1, 20, 64))
        linear_162 = None
        query_layer_81 = view_108.transpose(1, 2)
        view_108 = None
        linear_163 = torch._C._nn.linear(
            hidden_states_ln_27,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_109 = linear_163.view((1, -1, 20, 64))
        linear_163 = None
        key_layer_54 = view_109.transpose(1, 2)
        view_109 = None
        linear_164 = torch._C._nn.linear(
            hidden_states_ln_27,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_27 = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_110 = linear_164.view((1, -1, 20, 64))
        linear_164 = None
        value_layer_27 = view_110.transpose(1, 2)
        view_110 = None
        query_layer_82 = query_layer_81 * 0.125
        query_layer_81 = None
        arange_27 = torch.arange(13, device=device(type="cuda", index=0))
        t_27 = arange_27.type_as(
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_27 = None
        freqs_27 = torch.outer(
            t_27,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_27 = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_81 = torch.cat((freqs_27, freqs_27), dim=-1)
        freqs_27 = None
        emb_27 = cat_81.to(device(type="cuda", index=0))
        cat_81 = None
        cos_81 = emb_27.cos()
        getitem_272 = cos_81[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_81 = None
        sin_81 = emb_27.sin()
        emb_27 = None
        getitem_273 = sin_81[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_81 = None
        cos_82 = getitem_272[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_82 = getitem_273[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_194 = query_layer_82 * cos_82
        cos_82 = None
        chunk_54 = query_layer_82.chunk(2, dim=-1)
        query_layer_82 = None
        x1_54 = chunk_54[0]
        x2_54 = chunk_54[1]
        chunk_54 = None
        neg_54 = -x2_54
        x2_54 = None
        cat_82 = torch.cat((neg_54, x1_54), dim=-1)
        neg_54 = x1_54 = None
        mul_195 = cat_82 * sin_82
        cat_82 = sin_82 = None
        add_163 = mul_194 + mul_195
        mul_194 = mul_195 = None
        query_layer_83 = add_163.to(dtype=torch.float32)
        add_163 = None
        cos_83 = getitem_272[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_83 = getitem_273[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_196 = key_layer_54 * cos_83
        cos_83 = None
        chunk_55 = key_layer_54.chunk(2, dim=-1)
        key_layer_54 = None
        x1_55 = chunk_55[0]
        x2_55 = chunk_55[1]
        chunk_55 = None
        neg_55 = -x2_55
        x2_55 = None
        cat_83 = torch.cat((neg_55, x1_55), dim=-1)
        neg_55 = x1_55 = None
        mul_197 = cat_83 * sin_83
        cat_83 = sin_83 = None
        add_164 = mul_196 + mul_197
        mul_196 = mul_197 = None
        key_layer_55 = add_164.to(dtype=torch.float32)
        add_164 = None
        transpose_111 = key_layer_55.transpose(-1, -2)
        key_layer_55 = None
        attention_scores_54 = torch.matmul(query_layer_83, transpose_111)
        query_layer_83 = transpose_111 = None
        attention_scores_55 = attention_scores_54 + extended_attention_mask_2
        attention_scores_54 = None
        attention_probs_54 = torch.nn.functional.softmax(attention_scores_55, dim=-1)
        attention_scores_55 = None
        attention_probs_55 = torch.nn.functional.dropout(
            attention_probs_54, 0.0, False, False
        )
        attention_probs_54 = None
        to_114 = attention_probs_55.to(torch.float32)
        attention_probs_55 = None
        context_layer_81 = torch.matmul(to_114, value_layer_27)
        to_114 = value_layer_27 = None
        permute_27 = context_layer_81.permute(0, 2, 1, 3)
        context_layer_81 = None
        context_layer_82 = permute_27.contiguous()
        permute_27 = None
        context_layer_83 = context_layer_82.view((1, 13, 1280))
        context_layer_82 = None
        hidden_states_216 = torch._C._nn.linear(
            context_layer_83,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_83 = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_217 = torch.nn.functional.dropout(
            hidden_states_216, 0.0, False, False
        )
        hidden_states_216 = None
        hidden_states_218 = hidden_states_217 + hidden_states_215
        hidden_states_217 = hidden_states_215 = None
        attention_output_ln_27 = torch.nn.functional.layer_norm(
            hidden_states_218,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_27_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_219 = torch._C._nn.linear(
            attention_output_ln_27,
            l_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_27 = l_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_198 = hidden_states_219 * 0.5
        truediv_29 = hidden_states_219 / 1.4142135623730951
        hidden_states_219 = None
        erf_27 = torch.erf(truediv_29)
        truediv_29 = None
        add_167 = 1.0 + erf_27
        erf_27 = None
        hidden_states_220 = mul_198 * add_167
        mul_198 = add_167 = None
        hidden_states_221 = torch._C._nn.linear(
            hidden_states_220,
            l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_220 = l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_222 = torch.nn.functional.dropout(
            hidden_states_221, 0.0, False, False
        )
        hidden_states_221 = None
        hidden_states_223 = hidden_states_222 + hidden_states_218
        hidden_states_222 = hidden_states_218 = None
        hidden_states_ln_28 = torch.nn.functional.layer_norm(
            hidden_states_223,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_168 = torch._C._nn.linear(
            hidden_states_ln_28,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_112 = linear_168.view((1, -1, 20, 64))
        linear_168 = None
        query_layer_84 = view_112.transpose(1, 2)
        view_112 = None
        linear_169 = torch._C._nn.linear(
            hidden_states_ln_28,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_113 = linear_169.view((1, -1, 20, 64))
        linear_169 = None
        key_layer_56 = view_113.transpose(1, 2)
        view_113 = None
        linear_170 = torch._C._nn.linear(
            hidden_states_ln_28,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_28 = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_114 = linear_170.view((1, -1, 20, 64))
        linear_170 = None
        value_layer_28 = view_114.transpose(1, 2)
        view_114 = None
        query_layer_85 = query_layer_84 * 0.125
        query_layer_84 = None
        arange_28 = torch.arange(13, device=device(type="cuda", index=0))
        t_28 = arange_28.type_as(
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_28 = None
        freqs_28 = torch.outer(
            t_28,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_28 = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_84 = torch.cat((freqs_28, freqs_28), dim=-1)
        freqs_28 = None
        emb_28 = cat_84.to(device(type="cuda", index=0))
        cat_84 = None
        cos_84 = emb_28.cos()
        getitem_282 = cos_84[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_84 = None
        sin_84 = emb_28.sin()
        emb_28 = None
        getitem_283 = sin_84[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_84 = None
        cos_85 = getitem_282[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_85 = getitem_283[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_201 = query_layer_85 * cos_85
        cos_85 = None
        chunk_56 = query_layer_85.chunk(2, dim=-1)
        query_layer_85 = None
        x1_56 = chunk_56[0]
        x2_56 = chunk_56[1]
        chunk_56 = None
        neg_56 = -x2_56
        x2_56 = None
        cat_85 = torch.cat((neg_56, x1_56), dim=-1)
        neg_56 = x1_56 = None
        mul_202 = cat_85 * sin_85
        cat_85 = sin_85 = None
        add_169 = mul_201 + mul_202
        mul_201 = mul_202 = None
        query_layer_86 = add_169.to(dtype=torch.float32)
        add_169 = None
        cos_86 = getitem_282[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_86 = getitem_283[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_203 = key_layer_56 * cos_86
        cos_86 = None
        chunk_57 = key_layer_56.chunk(2, dim=-1)
        key_layer_56 = None
        x1_57 = chunk_57[0]
        x2_57 = chunk_57[1]
        chunk_57 = None
        neg_57 = -x2_57
        x2_57 = None
        cat_86 = torch.cat((neg_57, x1_57), dim=-1)
        neg_57 = x1_57 = None
        mul_204 = cat_86 * sin_86
        cat_86 = sin_86 = None
        add_170 = mul_203 + mul_204
        mul_203 = mul_204 = None
        key_layer_57 = add_170.to(dtype=torch.float32)
        add_170 = None
        transpose_115 = key_layer_57.transpose(-1, -2)
        key_layer_57 = None
        attention_scores_56 = torch.matmul(query_layer_86, transpose_115)
        query_layer_86 = transpose_115 = None
        attention_scores_57 = attention_scores_56 + extended_attention_mask_2
        attention_scores_56 = None
        attention_probs_56 = torch.nn.functional.softmax(attention_scores_57, dim=-1)
        attention_scores_57 = None
        attention_probs_57 = torch.nn.functional.dropout(
            attention_probs_56, 0.0, False, False
        )
        attention_probs_56 = None
        to_118 = attention_probs_57.to(torch.float32)
        attention_probs_57 = None
        context_layer_84 = torch.matmul(to_118, value_layer_28)
        to_118 = value_layer_28 = None
        permute_28 = context_layer_84.permute(0, 2, 1, 3)
        context_layer_84 = None
        context_layer_85 = permute_28.contiguous()
        permute_28 = None
        context_layer_86 = context_layer_85.view((1, 13, 1280))
        context_layer_85 = None
        hidden_states_224 = torch._C._nn.linear(
            context_layer_86,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_86 = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_225 = torch.nn.functional.dropout(
            hidden_states_224, 0.0, False, False
        )
        hidden_states_224 = None
        hidden_states_226 = hidden_states_225 + hidden_states_223
        hidden_states_225 = hidden_states_223 = None
        attention_output_ln_28 = torch.nn.functional.layer_norm(
            hidden_states_226,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_28_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_227 = torch._C._nn.linear(
            attention_output_ln_28,
            l_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_28 = l_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_205 = hidden_states_227 * 0.5
        truediv_30 = hidden_states_227 / 1.4142135623730951
        hidden_states_227 = None
        erf_28 = torch.erf(truediv_30)
        truediv_30 = None
        add_173 = 1.0 + erf_28
        erf_28 = None
        hidden_states_228 = mul_205 * add_173
        mul_205 = add_173 = None
        hidden_states_229 = torch._C._nn.linear(
            hidden_states_228,
            l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_228 = l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_230 = torch.nn.functional.dropout(
            hidden_states_229, 0.0, False, False
        )
        hidden_states_229 = None
        hidden_states_231 = hidden_states_230 + hidden_states_226
        hidden_states_230 = hidden_states_226 = None
        hidden_states_ln_29 = torch.nn.functional.layer_norm(
            hidden_states_231,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_174 = torch._C._nn.linear(
            hidden_states_ln_29,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_116 = linear_174.view((1, -1, 20, 64))
        linear_174 = None
        query_layer_87 = view_116.transpose(1, 2)
        view_116 = None
        linear_175 = torch._C._nn.linear(
            hidden_states_ln_29,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_117 = linear_175.view((1, -1, 20, 64))
        linear_175 = None
        key_layer_58 = view_117.transpose(1, 2)
        view_117 = None
        linear_176 = torch._C._nn.linear(
            hidden_states_ln_29,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_29 = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_118 = linear_176.view((1, -1, 20, 64))
        linear_176 = None
        value_layer_29 = view_118.transpose(1, 2)
        view_118 = None
        query_layer_88 = query_layer_87 * 0.125
        query_layer_87 = None
        arange_29 = torch.arange(13, device=device(type="cuda", index=0))
        t_29 = arange_29.type_as(
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_29 = None
        freqs_29 = torch.outer(
            t_29,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_29 = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_87 = torch.cat((freqs_29, freqs_29), dim=-1)
        freqs_29 = None
        emb_29 = cat_87.to(device(type="cuda", index=0))
        cat_87 = None
        cos_87 = emb_29.cos()
        getitem_292 = cos_87[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_87 = None
        sin_87 = emb_29.sin()
        emb_29 = None
        getitem_293 = sin_87[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_87 = None
        cos_88 = getitem_292[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_88 = getitem_293[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_208 = query_layer_88 * cos_88
        cos_88 = None
        chunk_58 = query_layer_88.chunk(2, dim=-1)
        query_layer_88 = None
        x1_58 = chunk_58[0]
        x2_58 = chunk_58[1]
        chunk_58 = None
        neg_58 = -x2_58
        x2_58 = None
        cat_88 = torch.cat((neg_58, x1_58), dim=-1)
        neg_58 = x1_58 = None
        mul_209 = cat_88 * sin_88
        cat_88 = sin_88 = None
        add_175 = mul_208 + mul_209
        mul_208 = mul_209 = None
        query_layer_89 = add_175.to(dtype=torch.float32)
        add_175 = None
        cos_89 = getitem_292[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_89 = getitem_293[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_210 = key_layer_58 * cos_89
        cos_89 = None
        chunk_59 = key_layer_58.chunk(2, dim=-1)
        key_layer_58 = None
        x1_59 = chunk_59[0]
        x2_59 = chunk_59[1]
        chunk_59 = None
        neg_59 = -x2_59
        x2_59 = None
        cat_89 = torch.cat((neg_59, x1_59), dim=-1)
        neg_59 = x1_59 = None
        mul_211 = cat_89 * sin_89
        cat_89 = sin_89 = None
        add_176 = mul_210 + mul_211
        mul_210 = mul_211 = None
        key_layer_59 = add_176.to(dtype=torch.float32)
        add_176 = None
        transpose_119 = key_layer_59.transpose(-1, -2)
        key_layer_59 = None
        attention_scores_58 = torch.matmul(query_layer_89, transpose_119)
        query_layer_89 = transpose_119 = None
        attention_scores_59 = attention_scores_58 + extended_attention_mask_2
        attention_scores_58 = None
        attention_probs_58 = torch.nn.functional.softmax(attention_scores_59, dim=-1)
        attention_scores_59 = None
        attention_probs_59 = torch.nn.functional.dropout(
            attention_probs_58, 0.0, False, False
        )
        attention_probs_58 = None
        to_122 = attention_probs_59.to(torch.float32)
        attention_probs_59 = None
        context_layer_87 = torch.matmul(to_122, value_layer_29)
        to_122 = value_layer_29 = None
        permute_29 = context_layer_87.permute(0, 2, 1, 3)
        context_layer_87 = None
        context_layer_88 = permute_29.contiguous()
        permute_29 = None
        context_layer_89 = context_layer_88.view((1, 13, 1280))
        context_layer_88 = None
        hidden_states_232 = torch._C._nn.linear(
            context_layer_89,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_89 = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_233 = torch.nn.functional.dropout(
            hidden_states_232, 0.0, False, False
        )
        hidden_states_232 = None
        hidden_states_234 = hidden_states_233 + hidden_states_231
        hidden_states_233 = hidden_states_231 = None
        attention_output_ln_29 = torch.nn.functional.layer_norm(
            hidden_states_234,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_29_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_235 = torch._C._nn.linear(
            attention_output_ln_29,
            l_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_29 = l_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_212 = hidden_states_235 * 0.5
        truediv_31 = hidden_states_235 / 1.4142135623730951
        hidden_states_235 = None
        erf_29 = torch.erf(truediv_31)
        truediv_31 = None
        add_179 = 1.0 + erf_29
        erf_29 = None
        hidden_states_236 = mul_212 * add_179
        mul_212 = add_179 = None
        hidden_states_237 = torch._C._nn.linear(
            hidden_states_236,
            l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_236 = l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_238 = torch.nn.functional.dropout(
            hidden_states_237, 0.0, False, False
        )
        hidden_states_237 = None
        hidden_states_239 = hidden_states_238 + hidden_states_234
        hidden_states_238 = hidden_states_234 = None
        hidden_states_ln_30 = torch.nn.functional.layer_norm(
            hidden_states_239,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_180 = torch._C._nn.linear(
            hidden_states_ln_30,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_120 = linear_180.view((1, -1, 20, 64))
        linear_180 = None
        query_layer_90 = view_120.transpose(1, 2)
        view_120 = None
        linear_181 = torch._C._nn.linear(
            hidden_states_ln_30,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_121 = linear_181.view((1, -1, 20, 64))
        linear_181 = None
        key_layer_60 = view_121.transpose(1, 2)
        view_121 = None
        linear_182 = torch._C._nn.linear(
            hidden_states_ln_30,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_30 = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_122 = linear_182.view((1, -1, 20, 64))
        linear_182 = None
        value_layer_30 = view_122.transpose(1, 2)
        view_122 = None
        query_layer_91 = query_layer_90 * 0.125
        query_layer_90 = None
        arange_30 = torch.arange(13, device=device(type="cuda", index=0))
        t_30 = arange_30.type_as(
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_30 = None
        freqs_30 = torch.outer(
            t_30,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_30 = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_90 = torch.cat((freqs_30, freqs_30), dim=-1)
        freqs_30 = None
        emb_30 = cat_90.to(device(type="cuda", index=0))
        cat_90 = None
        cos_90 = emb_30.cos()
        getitem_302 = cos_90[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_90 = None
        sin_90 = emb_30.sin()
        emb_30 = None
        getitem_303 = sin_90[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_90 = None
        cos_91 = getitem_302[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_91 = getitem_303[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_215 = query_layer_91 * cos_91
        cos_91 = None
        chunk_60 = query_layer_91.chunk(2, dim=-1)
        query_layer_91 = None
        x1_60 = chunk_60[0]
        x2_60 = chunk_60[1]
        chunk_60 = None
        neg_60 = -x2_60
        x2_60 = None
        cat_91 = torch.cat((neg_60, x1_60), dim=-1)
        neg_60 = x1_60 = None
        mul_216 = cat_91 * sin_91
        cat_91 = sin_91 = None
        add_181 = mul_215 + mul_216
        mul_215 = mul_216 = None
        query_layer_92 = add_181.to(dtype=torch.float32)
        add_181 = None
        cos_92 = getitem_302[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_92 = getitem_303[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_217 = key_layer_60 * cos_92
        cos_92 = None
        chunk_61 = key_layer_60.chunk(2, dim=-1)
        key_layer_60 = None
        x1_61 = chunk_61[0]
        x2_61 = chunk_61[1]
        chunk_61 = None
        neg_61 = -x2_61
        x2_61 = None
        cat_92 = torch.cat((neg_61, x1_61), dim=-1)
        neg_61 = x1_61 = None
        mul_218 = cat_92 * sin_92
        cat_92 = sin_92 = None
        add_182 = mul_217 + mul_218
        mul_217 = mul_218 = None
        key_layer_61 = add_182.to(dtype=torch.float32)
        add_182 = None
        transpose_123 = key_layer_61.transpose(-1, -2)
        key_layer_61 = None
        attention_scores_60 = torch.matmul(query_layer_92, transpose_123)
        query_layer_92 = transpose_123 = None
        attention_scores_61 = attention_scores_60 + extended_attention_mask_2
        attention_scores_60 = None
        attention_probs_60 = torch.nn.functional.softmax(attention_scores_61, dim=-1)
        attention_scores_61 = None
        attention_probs_61 = torch.nn.functional.dropout(
            attention_probs_60, 0.0, False, False
        )
        attention_probs_60 = None
        to_126 = attention_probs_61.to(torch.float32)
        attention_probs_61 = None
        context_layer_90 = torch.matmul(to_126, value_layer_30)
        to_126 = value_layer_30 = None
        permute_30 = context_layer_90.permute(0, 2, 1, 3)
        context_layer_90 = None
        context_layer_91 = permute_30.contiguous()
        permute_30 = None
        context_layer_92 = context_layer_91.view((1, 13, 1280))
        context_layer_91 = None
        hidden_states_240 = torch._C._nn.linear(
            context_layer_92,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_92 = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_241 = torch.nn.functional.dropout(
            hidden_states_240, 0.0, False, False
        )
        hidden_states_240 = None
        hidden_states_242 = hidden_states_241 + hidden_states_239
        hidden_states_241 = hidden_states_239 = None
        attention_output_ln_30 = torch.nn.functional.layer_norm(
            hidden_states_242,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_30_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_243 = torch._C._nn.linear(
            attention_output_ln_30,
            l_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_30 = l_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_219 = hidden_states_243 * 0.5
        truediv_32 = hidden_states_243 / 1.4142135623730951
        hidden_states_243 = None
        erf_30 = torch.erf(truediv_32)
        truediv_32 = None
        add_185 = 1.0 + erf_30
        erf_30 = None
        hidden_states_244 = mul_219 * add_185
        mul_219 = add_185 = None
        hidden_states_245 = torch._C._nn.linear(
            hidden_states_244,
            l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_244 = l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_246 = torch.nn.functional.dropout(
            hidden_states_245, 0.0, False, False
        )
        hidden_states_245 = None
        hidden_states_247 = hidden_states_246 + hidden_states_242
        hidden_states_246 = hidden_states_242 = None
        hidden_states_ln_31 = torch.nn.functional.layer_norm(
            hidden_states_247,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_186 = torch._C._nn.linear(
            hidden_states_ln_31,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_124 = linear_186.view((1, -1, 20, 64))
        linear_186 = None
        query_layer_93 = view_124.transpose(1, 2)
        view_124 = None
        linear_187 = torch._C._nn.linear(
            hidden_states_ln_31,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_125 = linear_187.view((1, -1, 20, 64))
        linear_187 = None
        key_layer_62 = view_125.transpose(1, 2)
        view_125 = None
        linear_188 = torch._C._nn.linear(
            hidden_states_ln_31,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_31 = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_126 = linear_188.view((1, -1, 20, 64))
        linear_188 = None
        value_layer_31 = view_126.transpose(1, 2)
        view_126 = None
        query_layer_94 = query_layer_93 * 0.125
        query_layer_93 = None
        arange_31 = torch.arange(13, device=device(type="cuda", index=0))
        t_31 = arange_31.type_as(
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_31 = None
        freqs_31 = torch.outer(
            t_31,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_31 = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_93 = torch.cat((freqs_31, freqs_31), dim=-1)
        freqs_31 = None
        emb_31 = cat_93.to(device(type="cuda", index=0))
        cat_93 = None
        cos_93 = emb_31.cos()
        getitem_312 = cos_93[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_93 = None
        sin_93 = emb_31.sin()
        emb_31 = None
        getitem_313 = sin_93[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_93 = None
        cos_94 = getitem_312[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_94 = getitem_313[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_222 = query_layer_94 * cos_94
        cos_94 = None
        chunk_62 = query_layer_94.chunk(2, dim=-1)
        query_layer_94 = None
        x1_62 = chunk_62[0]
        x2_62 = chunk_62[1]
        chunk_62 = None
        neg_62 = -x2_62
        x2_62 = None
        cat_94 = torch.cat((neg_62, x1_62), dim=-1)
        neg_62 = x1_62 = None
        mul_223 = cat_94 * sin_94
        cat_94 = sin_94 = None
        add_187 = mul_222 + mul_223
        mul_222 = mul_223 = None
        query_layer_95 = add_187.to(dtype=torch.float32)
        add_187 = None
        cos_95 = getitem_312[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_95 = getitem_313[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_224 = key_layer_62 * cos_95
        cos_95 = None
        chunk_63 = key_layer_62.chunk(2, dim=-1)
        key_layer_62 = None
        x1_63 = chunk_63[0]
        x2_63 = chunk_63[1]
        chunk_63 = None
        neg_63 = -x2_63
        x2_63 = None
        cat_95 = torch.cat((neg_63, x1_63), dim=-1)
        neg_63 = x1_63 = None
        mul_225 = cat_95 * sin_95
        cat_95 = sin_95 = None
        add_188 = mul_224 + mul_225
        mul_224 = mul_225 = None
        key_layer_63 = add_188.to(dtype=torch.float32)
        add_188 = None
        transpose_127 = key_layer_63.transpose(-1, -2)
        key_layer_63 = None
        attention_scores_62 = torch.matmul(query_layer_95, transpose_127)
        query_layer_95 = transpose_127 = None
        attention_scores_63 = attention_scores_62 + extended_attention_mask_2
        attention_scores_62 = None
        attention_probs_62 = torch.nn.functional.softmax(attention_scores_63, dim=-1)
        attention_scores_63 = None
        attention_probs_63 = torch.nn.functional.dropout(
            attention_probs_62, 0.0, False, False
        )
        attention_probs_62 = None
        to_130 = attention_probs_63.to(torch.float32)
        attention_probs_63 = None
        context_layer_93 = torch.matmul(to_130, value_layer_31)
        to_130 = value_layer_31 = None
        permute_31 = context_layer_93.permute(0, 2, 1, 3)
        context_layer_93 = None
        context_layer_94 = permute_31.contiguous()
        permute_31 = None
        context_layer_95 = context_layer_94.view((1, 13, 1280))
        context_layer_94 = None
        hidden_states_248 = torch._C._nn.linear(
            context_layer_95,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_95 = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_249 = torch.nn.functional.dropout(
            hidden_states_248, 0.0, False, False
        )
        hidden_states_248 = None
        hidden_states_250 = hidden_states_249 + hidden_states_247
        hidden_states_249 = hidden_states_247 = None
        attention_output_ln_31 = torch.nn.functional.layer_norm(
            hidden_states_250,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_31_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_251 = torch._C._nn.linear(
            attention_output_ln_31,
            l_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_31 = l_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_226 = hidden_states_251 * 0.5
        truediv_33 = hidden_states_251 / 1.4142135623730951
        hidden_states_251 = None
        erf_31 = torch.erf(truediv_33)
        truediv_33 = None
        add_191 = 1.0 + erf_31
        erf_31 = None
        hidden_states_252 = mul_226 * add_191
        mul_226 = add_191 = None
        hidden_states_253 = torch._C._nn.linear(
            hidden_states_252,
            l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_252 = l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_254 = torch.nn.functional.dropout(
            hidden_states_253, 0.0, False, False
        )
        hidden_states_253 = None
        hidden_states_255 = hidden_states_254 + hidden_states_250
        hidden_states_254 = hidden_states_250 = None
        hidden_states_ln_32 = torch.nn.functional.layer_norm(
            hidden_states_255,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_layer_norm_parameters_bias_ = (None)
        linear_192 = torch._C._nn.linear(
            hidden_states_ln_32,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_128 = linear_192.view((1, -1, 20, 64))
        linear_192 = None
        query_layer_96 = view_128.transpose(1, 2)
        view_128 = None
        linear_193 = torch._C._nn.linear(
            hidden_states_ln_32,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_129 = linear_193.view((1, -1, 20, 64))
        linear_193 = None
        key_layer_64 = view_129.transpose(1, 2)
        view_129 = None
        linear_194 = torch._C._nn.linear(
            hidden_states_ln_32,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_ln_32 = l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_130 = linear_194.view((1, -1, 20, 64))
        linear_194 = None
        value_layer_32 = view_130.transpose(1, 2)
        view_130 = None
        query_layer_97 = query_layer_96 * 0.125
        query_layer_96 = None
        arange_32 = torch.arange(13, device=device(type="cuda", index=0))
        t_32 = arange_32.type_as(
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_
        )
        arange_32 = None
        freqs_32 = torch.outer(
            t_32,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_,
        )
        t_32 = l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_ = (None)
        cat_96 = torch.cat((freqs_32, freqs_32), dim=-1)
        freqs_32 = None
        emb_32 = cat_96.to(device(type="cuda", index=0))
        cat_96 = None
        cos_96 = emb_32.cos()
        getitem_322 = cos_96[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        cos_96 = None
        sin_96 = emb_32.sin()
        emb_32 = None
        getitem_323 = sin_96[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        sin_96 = None
        cos_97 = getitem_322[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_97 = getitem_323[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_229 = query_layer_97 * cos_97
        cos_97 = None
        chunk_64 = query_layer_97.chunk(2, dim=-1)
        query_layer_97 = None
        x1_64 = chunk_64[0]
        x2_64 = chunk_64[1]
        chunk_64 = None
        neg_64 = -x2_64
        x2_64 = None
        cat_97 = torch.cat((neg_64, x1_64), dim=-1)
        neg_64 = x1_64 = None
        mul_230 = cat_97 * sin_97
        cat_97 = sin_97 = None
        add_193 = mul_229 + mul_230
        mul_229 = mul_230 = None
        query_layer_98 = add_193.to(dtype=torch.float32)
        add_193 = None
        cos_98 = getitem_322[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        sin_98 = getitem_323[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 13, None),
                slice(None, None, None),
            )
        ]
        mul_231 = key_layer_64 * cos_98
        cos_98 = None
        chunk_65 = key_layer_64.chunk(2, dim=-1)
        key_layer_64 = None
        x1_65 = chunk_65[0]
        x2_65 = chunk_65[1]
        chunk_65 = None
        neg_65 = -x2_65
        x2_65 = None
        cat_98 = torch.cat((neg_65, x1_65), dim=-1)
        neg_65 = x1_65 = None
        mul_232 = cat_98 * sin_98
        cat_98 = sin_98 = None
        add_194 = mul_231 + mul_232
        mul_231 = mul_232 = None
        key_layer_65 = add_194.to(dtype=torch.float32)
        add_194 = None
        transpose_131 = key_layer_65.transpose(-1, -2)
        key_layer_65 = None
        attention_scores_64 = torch.matmul(query_layer_98, transpose_131)
        query_layer_98 = transpose_131 = None
        attention_scores_65 = attention_scores_64 + extended_attention_mask_2
        attention_scores_64 = extended_attention_mask_2 = None
        attention_probs_64 = torch.nn.functional.softmax(attention_scores_65, dim=-1)
        attention_scores_65 = None
        attention_probs_65 = torch.nn.functional.dropout(
            attention_probs_64, 0.0, False, False
        )
        attention_probs_64 = None
        to_134 = attention_probs_65.to(torch.float32)
        attention_probs_65 = None
        context_layer_96 = torch.matmul(to_134, value_layer_32)
        to_134 = value_layer_32 = None
        permute_32 = context_layer_96.permute(0, 2, 1, 3)
        context_layer_96 = None
        context_layer_97 = permute_32.contiguous()
        permute_32 = None
        context_layer_98 = context_layer_97.view((1, 13, 1280))
        context_layer_97 = None
        hidden_states_256 = torch._C._nn.linear(
            context_layer_98,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_98 = l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_257 = torch.nn.functional.dropout(
            hidden_states_256, 0.0, False, False
        )
        hidden_states_256 = None
        hidden_states_258 = hidden_states_257 + hidden_states_255
        hidden_states_257 = hidden_states_255 = None
        attention_output_ln_32 = torch.nn.functional.layer_norm(
            hidden_states_258,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_32_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_layer_modules_32_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_layer_norm_parameters_bias_ = (None)
        hidden_states_259 = torch._C._nn.linear(
            attention_output_ln_32,
            l_self_modules_encoder_modules_layer_modules_32_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_intermediate_modules_dense_parameters_bias_,
        )
        attention_output_ln_32 = l_self_modules_encoder_modules_layer_modules_32_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_intermediate_modules_dense_parameters_bias_ = (None)
        mul_233 = hidden_states_259 * 0.5
        truediv_34 = hidden_states_259 / 1.4142135623730951
        hidden_states_259 = None
        erf_32 = torch.erf(truediv_34)
        truediv_34 = None
        add_197 = 1.0 + erf_32
        erf_32 = None
        hidden_states_260 = mul_233 * add_197
        mul_233 = add_197 = None
        hidden_states_261 = torch._C._nn.linear(
            hidden_states_260,
            l_self_modules_encoder_modules_layer_modules_32_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_32_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_260 = l_self_modules_encoder_modules_layer_modules_32_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_32_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_262 = torch.nn.functional.dropout(
            hidden_states_261, 0.0, False, False
        )
        hidden_states_261 = None
        hidden_states_263 = hidden_states_262 + hidden_states_258
        hidden_states_262 = hidden_states_258 = None
        hidden_states_264 = torch.nn.functional.layer_norm(
            hidden_states_263,
            (1280,),
            l_self_modules_encoder_modules_emb_layer_norm_after_parameters_weight_,
            l_self_modules_encoder_modules_emb_layer_norm_after_parameters_bias_,
            1e-05,
        )
        hidden_states_263 = (
            l_self_modules_encoder_modules_emb_layer_norm_after_parameters_weight_
        ) = l_self_modules_encoder_modules_emb_layer_norm_after_parameters_bias_ = None
        first_token_tensor = hidden_states_264[(slice(None, None, None), 0)]
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
        return (
            getitem_3,
            getitem_2,
            getitem_13,
            getitem_12,
            getitem_23,
            getitem_22,
            getitem_33,
            getitem_32,
            getitem_43,
            getitem_42,
            getitem_53,
            getitem_52,
            getitem_63,
            getitem_62,
            getitem_73,
            getitem_72,
            getitem_83,
            getitem_82,
            getitem_93,
            getitem_92,
            getitem_103,
            getitem_102,
            getitem_113,
            getitem_112,
            getitem_123,
            getitem_122,
            getitem_133,
            getitem_132,
            getitem_143,
            getitem_142,
            getitem_153,
            getitem_152,
            getitem_163,
            getitem_162,
            getitem_173,
            getitem_172,
            getitem_183,
            getitem_182,
            getitem_193,
            getitem_192,
            getitem_203,
            getitem_202,
            getitem_213,
            getitem_212,
            getitem_223,
            getitem_222,
            getitem_233,
            getitem_232,
            getitem_243,
            getitem_242,
            getitem_253,
            getitem_252,
            getitem_263,
            getitem_262,
            getitem_273,
            getitem_272,
            getitem_283,
            getitem_282,
            getitem_293,
            getitem_292,
            getitem_303,
            getitem_302,
            getitem_313,
            getitem_312,
            getitem_323,
            getitem_322,
            hidden_states_264,
            pooled_output_1,
        )
