import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s27: torch.SymInt,
        L_kwargs_input_ids_: torch.Tensor,
        L_self_modules_bert_modules_embeddings_buffers_position_ids_: torch.Tensor,
        L_kwargs_token_type_ids_: torch.Tensor,
        L_self_modules_bert_modules_embeddings_modules_word_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_embeddings_modules_word_embeddings_norm_type: torch.Tensor,
        L_self_modules_bert_modules_embeddings_modules_token_type_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_embeddings_modules_token_type_embeddings_norm_type: torch.Tensor,
        L_self_modules_bert_modules_embeddings_modules_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_embeddings_modules_position_embeddings_norm_type: torch.Tensor,
        L_self_modules_bert_modules_embeddings_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_embeddings_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_embeddings_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_embeddings_modules_dropout_p: torch.Tensor,
        L_kwargs_attention_mask_: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_scaling: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_scaling: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_scaling: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_scaling: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_scaling: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_scaling: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_scaling: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_scaling: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_scaling: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_scaling: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_scaling: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_scaling: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_eps: torch.Tensor,
        L_self_modules_dropout_p: torch.Tensor,
        L_self_modules_classifier_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_kwargs_input_ids_ = L_kwargs_input_ids_
        l_self_modules_bert_modules_embeddings_buffers_position_ids_ = (
            L_self_modules_bert_modules_embeddings_buffers_position_ids_
        )
        l_kwargs_token_type_ids_ = L_kwargs_token_type_ids_
        l_self_modules_bert_modules_embeddings_modules_word_embeddings_parameters_weight_ = L_self_modules_bert_modules_embeddings_modules_word_embeddings_parameters_weight_
        l_self_modules_bert_modules_embeddings_modules_word_embeddings_norm_type = (
            L_self_modules_bert_modules_embeddings_modules_word_embeddings_norm_type
        )
        l_self_modules_bert_modules_embeddings_modules_token_type_embeddings_parameters_weight_ = L_self_modules_bert_modules_embeddings_modules_token_type_embeddings_parameters_weight_
        l_self_modules_bert_modules_embeddings_modules_token_type_embeddings_norm_type = L_self_modules_bert_modules_embeddings_modules_token_type_embeddings_norm_type
        l_self_modules_bert_modules_embeddings_modules_position_embeddings_parameters_weight_ = L_self_modules_bert_modules_embeddings_modules_position_embeddings_parameters_weight_
        l_self_modules_bert_modules_embeddings_modules_position_embeddings_norm_type = (
            L_self_modules_bert_modules_embeddings_modules_position_embeddings_norm_type
        )
        l_self_modules_bert_modules_embeddings_modules_layer_norm_parameters_weight_ = (
            L_self_modules_bert_modules_embeddings_modules_LayerNorm_parameters_weight_
        )
        l_self_modules_bert_modules_embeddings_modules_layer_norm_parameters_bias_ = (
            L_self_modules_bert_modules_embeddings_modules_LayerNorm_parameters_bias_
        )
        l_self_modules_bert_modules_embeddings_modules_layer_norm_eps = (
            L_self_modules_bert_modules_embeddings_modules_LayerNorm_eps
        )
        l_self_modules_bert_modules_embeddings_modules_dropout_p = (
            L_self_modules_bert_modules_embeddings_modules_dropout_p
        )
        l_kwargs_attention_mask_ = L_kwargs_attention_mask_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_scaling = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_scaling
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_scaling = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_scaling
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_scaling = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_scaling
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_scaling = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_scaling
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_scaling = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_scaling
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_scaling = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_scaling
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_scaling = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_scaling
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_scaling = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_scaling
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_scaling = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_scaling
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_scaling = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_scaling
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_scaling = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_scaling
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_scaling = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_scaling
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_eps
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_weight_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_bias_ = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_eps = L_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_eps
        l_self_modules_dropout_p = L_self_modules_dropout_p
        l_self_modules_classifier_parameters_weight_ = (
            L_self_modules_classifier_parameters_weight_
        )
        l_self_modules_classifier_parameters_bias_ = (
            L_self_modules_classifier_parameters_bias_
        )
        cache_position = torch.arange(0, s27, device=device(type="cpu"))
        cache_position = None
        position_ids = l_self_modules_bert_modules_embeddings_buffers_position_ids_[
            (slice(None, None, None), slice(0, s27, None))
        ]
        l_self_modules_bert_modules_embeddings_buffers_position_ids_ = None
        item = (
            l_self_modules_bert_modules_embeddings_modules_word_embeddings_norm_type.item()
        )
        l_self_modules_bert_modules_embeddings_modules_word_embeddings_norm_type = None
        inputs_embeds = torch.nn.functional.embedding(
            l_kwargs_input_ids_,
            l_self_modules_bert_modules_embeddings_modules_word_embeddings_parameters_weight_,
            0,
            None,
            item,
            False,
            False,
        )
        l_kwargs_input_ids_ = l_self_modules_bert_modules_embeddings_modules_word_embeddings_parameters_weight_ = (item) = (
            None
        )
        item_1 = (
            l_self_modules_bert_modules_embeddings_modules_token_type_embeddings_norm_type.item()
        )
        l_self_modules_bert_modules_embeddings_modules_token_type_embeddings_norm_type = (
            None
        )
        token_type_embeddings = torch.nn.functional.embedding(
            l_kwargs_token_type_ids_,
            l_self_modules_bert_modules_embeddings_modules_token_type_embeddings_parameters_weight_,
            None,
            None,
            item_1,
            False,
            False,
        )
        l_kwargs_token_type_ids_ = l_self_modules_bert_modules_embeddings_modules_token_type_embeddings_parameters_weight_ = (item_1) = (
            None
        )
        embeddings = inputs_embeds + token_type_embeddings
        inputs_embeds = token_type_embeddings = None
        item_2 = (
            l_self_modules_bert_modules_embeddings_modules_position_embeddings_norm_type.item()
        )
        l_self_modules_bert_modules_embeddings_modules_position_embeddings_norm_type = (
            None
        )
        position_embeddings = torch.nn.functional.embedding(
            position_ids,
            l_self_modules_bert_modules_embeddings_modules_position_embeddings_parameters_weight_,
            None,
            None,
            item_2,
            False,
            False,
        )
        position_ids = l_self_modules_bert_modules_embeddings_modules_position_embeddings_parameters_weight_ = (item_2) = (
            None
        )
        embeddings_1 = embeddings + position_embeddings
        embeddings = position_embeddings = None
        item_3 = l_self_modules_bert_modules_embeddings_modules_layer_norm_eps.item()
        l_self_modules_bert_modules_embeddings_modules_layer_norm_eps = None
        embeddings_2 = torch.nn.functional.layer_norm(
            embeddings_1,
            (768,),
            l_self_modules_bert_modules_embeddings_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_embeddings_modules_layer_norm_parameters_bias_,
            item_3,
        )
        embeddings_1 = (
            l_self_modules_bert_modules_embeddings_modules_layer_norm_parameters_weight_
        ) = (
            l_self_modules_bert_modules_embeddings_modules_layer_norm_parameters_bias_
        ) = item_3 = None
        item_4 = l_self_modules_bert_modules_embeddings_modules_dropout_p.item()
        l_self_modules_bert_modules_embeddings_modules_dropout_p = None
        embeddings_3 = torch.nn.functional.dropout(embeddings_2, item_4, False, False)
        embeddings_2 = item_4 = None
        cache_position_1 = torch.arange(
            s27, device=device(type="cpu"), dtype=torch.int64
        )
        attention_mask = l_kwargs_attention_mask_.to(
            device=device(type="cpu"), dtype=torch.bool
        )
        l_kwargs_attention_mask_ = None
        batch_arange = torch.arange(1, device=device(type="cpu"))
        head_arange = torch.arange(1, device=device(type="cpu"))
        head_arange = None
        arange_4 = torch.arange(s27, device=device(type="cpu"))
        kv_arange = arange_4 + 0
        arange_4 = None
        batch_indices = batch_arange[(slice(None, None, None), None, None, None)]
        batch_arange = None
        q_indices = cache_position_1[(None, None, slice(None, None, None), None)]
        cache_position_1 = None
        kv_indices = kv_arange[(None, None, None, slice(None, None, None))]
        kv_arange = None
        result = q_indices.new_ones((), dtype=torch.bool)
        ge = q_indices >= 0
        q_indices = None
        to_1 = ge.to(device(type="cpu"))
        ge = None
        result_1 = result & to_1
        result = to_1 = None
        getitem_22 = attention_mask[(batch_indices, kv_indices)]
        attention_mask = batch_indices = kv_indices = None
        to_2 = getitem_22.to(device(type="cpu"))
        getitem_22 = None
        result_2 = result_1 & to_2
        result_1 = to_2 = None
        attention_mask_1 = result_2.expand(1, -1, s27, s27)
        result_2 = None
        linear = torch.nn.functional.linear(
            embeddings_3,
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view = linear.view(1, s27, -1, 64)
        linear = None
        query_layer = view.transpose(1, 2)
        view = None
        linear_1 = torch.nn.functional.linear(
            embeddings_3,
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_1 = linear_1.view(1, s27, -1, 64)
        linear_1 = None
        key_layer = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch.nn.functional.linear(
            embeddings_3,
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_2 = linear_2.view(1, s27, -1, 64)
        linear_2 = None
        value_layer = view_2.transpose(1, 2)
        view_2 = None
        item_5 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_scaling.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_scaling = (
            None
        )
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=item_5,
            is_causal=False,
        )
        query_layer = key_layer = value_layer = item_5 = None
        transpose_3 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_3.contiguous()
        transpose_3 = None
        reshape = attn_output_1.reshape(1, s27, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        hidden_states = torch.nn.functional.linear(
            attn_output_2,
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_2 = l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_6 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_1 = torch.nn.functional.dropout(
            hidden_states, item_6, False, False
        )
        hidden_states = item_6 = None
        add_6 = hidden_states_1 + embeddings_3
        hidden_states_1 = embeddings_3 = None
        item_7 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_2 = torch.nn.functional.layer_norm(
            add_6,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            item_7,
        )
        add_6 = l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (item_7) = (
            None
        )
        hidden_states_3 = torch.nn.functional.linear(
            hidden_states_2,
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_4 = torch.nn.functional.gelu(hidden_states_3)
        hidden_states_3 = None
        hidden_states_5 = torch.nn.functional.linear(
            hidden_states_4,
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_4 = l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        item_8 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, item_8, False, False
        )
        hidden_states_5 = item_8 = None
        add_7 = hidden_states_6 + hidden_states_2
        hidden_states_6 = hidden_states_2 = None
        item_9 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_7 = torch.nn.functional.layer_norm(
            add_7,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_,
            item_9,
        )
        add_7 = l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_0_modules_output_modules_layer_norm_parameters_bias_ = (item_9) = (
            None
        )
        linear_6 = torch.nn.functional.linear(
            hidden_states_7,
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_3 = linear_6.view(1, s27, -1, 64)
        linear_6 = None
        query_layer_1 = view_3.transpose(1, 2)
        view_3 = None
        linear_7 = torch.nn.functional.linear(
            hidden_states_7,
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_4 = linear_7.view(1, s27, -1, 64)
        linear_7 = None
        key_layer_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_8 = torch.nn.functional.linear(
            hidden_states_7,
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_5 = linear_8.view(1, s27, -1, 64)
        linear_8 = None
        value_layer_1 = view_5.transpose(1, 2)
        view_5 = None
        item_10 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_scaling.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_scaling = (
            None
        )
        attn_output_3 = torch.nn.functional.scaled_dot_product_attention(
            query_layer_1,
            key_layer_1,
            value_layer_1,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=item_10,
            is_causal=False,
        )
        query_layer_1 = key_layer_1 = value_layer_1 = item_10 = None
        transpose_7 = attn_output_3.transpose(1, 2)
        attn_output_3 = None
        attn_output_4 = transpose_7.contiguous()
        transpose_7 = None
        reshape_1 = attn_output_4.reshape(1, s27, -1)
        attn_output_4 = None
        attn_output_5 = reshape_1.contiguous()
        reshape_1 = None
        hidden_states_8 = torch.nn.functional.linear(
            attn_output_5,
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_5 = l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_11 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, item_11, False, False
        )
        hidden_states_8 = item_11 = None
        add_8 = hidden_states_9 + hidden_states_7
        hidden_states_9 = hidden_states_7 = None
        item_12 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_10 = torch.nn.functional.layer_norm(
            add_8,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            item_12,
        )
        add_8 = l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (item_12) = (
            None
        )
        hidden_states_11 = torch.nn.functional.linear(
            hidden_states_10,
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_12 = torch.nn.functional.gelu(hidden_states_11)
        hidden_states_11 = None
        hidden_states_13 = torch.nn.functional.linear(
            hidden_states_12,
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_12 = l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        item_13 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_14 = torch.nn.functional.dropout(
            hidden_states_13, item_13, False, False
        )
        hidden_states_13 = item_13 = None
        add_9 = hidden_states_14 + hidden_states_10
        hidden_states_14 = hidden_states_10 = None
        item_14 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_15 = torch.nn.functional.layer_norm(
            add_9,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_,
            item_14,
        )
        add_9 = l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_1_modules_output_modules_layer_norm_parameters_bias_ = (item_14) = (
            None
        )
        linear_12 = torch.nn.functional.linear(
            hidden_states_15,
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_6 = linear_12.view(1, s27, -1, 64)
        linear_12 = None
        query_layer_2 = view_6.transpose(1, 2)
        view_6 = None
        linear_13 = torch.nn.functional.linear(
            hidden_states_15,
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_7 = linear_13.view(1, s27, -1, 64)
        linear_13 = None
        key_layer_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_14 = torch.nn.functional.linear(
            hidden_states_15,
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_8 = linear_14.view(1, s27, -1, 64)
        linear_14 = None
        value_layer_2 = view_8.transpose(1, 2)
        view_8 = None
        item_15 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_scaling.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_scaling = (
            None
        )
        attn_output_6 = torch.nn.functional.scaled_dot_product_attention(
            query_layer_2,
            key_layer_2,
            value_layer_2,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=item_15,
            is_causal=False,
        )
        query_layer_2 = key_layer_2 = value_layer_2 = item_15 = None
        transpose_11 = attn_output_6.transpose(1, 2)
        attn_output_6 = None
        attn_output_7 = transpose_11.contiguous()
        transpose_11 = None
        reshape_2 = attn_output_7.reshape(1, s27, -1)
        attn_output_7 = None
        attn_output_8 = reshape_2.contiguous()
        reshape_2 = None
        hidden_states_16 = torch.nn.functional.linear(
            attn_output_8,
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_8 = l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_16 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_17 = torch.nn.functional.dropout(
            hidden_states_16, item_16, False, False
        )
        hidden_states_16 = item_16 = None
        add_10 = hidden_states_17 + hidden_states_15
        hidden_states_17 = hidden_states_15 = None
        item_17 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_18 = torch.nn.functional.layer_norm(
            add_10,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            item_17,
        )
        add_10 = l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (item_17) = (
            None
        )
        hidden_states_19 = torch.nn.functional.linear(
            hidden_states_18,
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_20 = torch.nn.functional.gelu(hidden_states_19)
        hidden_states_19 = None
        hidden_states_21 = torch.nn.functional.linear(
            hidden_states_20,
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        item_18 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, item_18, False, False
        )
        hidden_states_21 = item_18 = None
        add_11 = hidden_states_22 + hidden_states_18
        hidden_states_22 = hidden_states_18 = None
        item_19 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_23 = torch.nn.functional.layer_norm(
            add_11,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_,
            item_19,
        )
        add_11 = l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_2_modules_output_modules_layer_norm_parameters_bias_ = (item_19) = (
            None
        )
        linear_18 = torch.nn.functional.linear(
            hidden_states_23,
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_9 = linear_18.view(1, s27, -1, 64)
        linear_18 = None
        query_layer_3 = view_9.transpose(1, 2)
        view_9 = None
        linear_19 = torch.nn.functional.linear(
            hidden_states_23,
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_10 = linear_19.view(1, s27, -1, 64)
        linear_19 = None
        key_layer_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_20 = torch.nn.functional.linear(
            hidden_states_23,
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_11 = linear_20.view(1, s27, -1, 64)
        linear_20 = None
        value_layer_3 = view_11.transpose(1, 2)
        view_11 = None
        item_20 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_scaling.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_scaling = (
            None
        )
        attn_output_9 = torch.nn.functional.scaled_dot_product_attention(
            query_layer_3,
            key_layer_3,
            value_layer_3,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=item_20,
            is_causal=False,
        )
        query_layer_3 = key_layer_3 = value_layer_3 = item_20 = None
        transpose_15 = attn_output_9.transpose(1, 2)
        attn_output_9 = None
        attn_output_10 = transpose_15.contiguous()
        transpose_15 = None
        reshape_3 = attn_output_10.reshape(1, s27, -1)
        attn_output_10 = None
        attn_output_11 = reshape_3.contiguous()
        reshape_3 = None
        hidden_states_24 = torch.nn.functional.linear(
            attn_output_11,
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_11 = l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_21 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, item_21, False, False
        )
        hidden_states_24 = item_21 = None
        add_12 = hidden_states_25 + hidden_states_23
        hidden_states_25 = hidden_states_23 = None
        item_22 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_26 = torch.nn.functional.layer_norm(
            add_12,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            item_22,
        )
        add_12 = l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (item_22) = (
            None
        )
        hidden_states_27 = torch.nn.functional.linear(
            hidden_states_26,
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_28 = torch.nn.functional.gelu(hidden_states_27)
        hidden_states_27 = None
        hidden_states_29 = torch.nn.functional.linear(
            hidden_states_28,
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_28 = l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = (None)
        item_23 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_30 = torch.nn.functional.dropout(
            hidden_states_29, item_23, False, False
        )
        hidden_states_29 = item_23 = None
        add_13 = hidden_states_30 + hidden_states_26
        hidden_states_30 = hidden_states_26 = None
        item_24 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_31 = torch.nn.functional.layer_norm(
            add_13,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_,
            item_24,
        )
        add_13 = l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_3_modules_output_modules_layer_norm_parameters_bias_ = (item_24) = (
            None
        )
        linear_24 = torch.nn.functional.linear(
            hidden_states_31,
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_12 = linear_24.view(1, s27, -1, 64)
        linear_24 = None
        query_layer_4 = view_12.transpose(1, 2)
        view_12 = None
        linear_25 = torch.nn.functional.linear(
            hidden_states_31,
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_13 = linear_25.view(1, s27, -1, 64)
        linear_25 = None
        key_layer_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_26 = torch.nn.functional.linear(
            hidden_states_31,
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_14 = linear_26.view(1, s27, -1, 64)
        linear_26 = None
        value_layer_4 = view_14.transpose(1, 2)
        view_14 = None
        item_25 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_scaling.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_scaling = (
            None
        )
        attn_output_12 = torch.nn.functional.scaled_dot_product_attention(
            query_layer_4,
            key_layer_4,
            value_layer_4,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=item_25,
            is_causal=False,
        )
        query_layer_4 = key_layer_4 = value_layer_4 = item_25 = None
        transpose_19 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_19.contiguous()
        transpose_19 = None
        reshape_4 = attn_output_13.reshape(1, s27, -1)
        attn_output_13 = None
        attn_output_14 = reshape_4.contiguous()
        reshape_4 = None
        hidden_states_32 = torch.nn.functional.linear(
            attn_output_14,
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_14 = l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_26 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_33 = torch.nn.functional.dropout(
            hidden_states_32, item_26, False, False
        )
        hidden_states_32 = item_26 = None
        add_14 = hidden_states_33 + hidden_states_31
        hidden_states_33 = hidden_states_31 = None
        item_27 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_34 = torch.nn.functional.layer_norm(
            add_14,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            item_27,
        )
        add_14 = l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (item_27) = (
            None
        )
        hidden_states_35 = torch.nn.functional.linear(
            hidden_states_34,
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_36 = torch.nn.functional.gelu(hidden_states_35)
        hidden_states_35 = None
        hidden_states_37 = torch.nn.functional.linear(
            hidden_states_36,
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_36 = l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = (None)
        item_28 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_38 = torch.nn.functional.dropout(
            hidden_states_37, item_28, False, False
        )
        hidden_states_37 = item_28 = None
        add_15 = hidden_states_38 + hidden_states_34
        hidden_states_38 = hidden_states_34 = None
        item_29 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_39 = torch.nn.functional.layer_norm(
            add_15,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_,
            item_29,
        )
        add_15 = l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_4_modules_output_modules_layer_norm_parameters_bias_ = (item_29) = (
            None
        )
        linear_30 = torch.nn.functional.linear(
            hidden_states_39,
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_15 = linear_30.view(1, s27, -1, 64)
        linear_30 = None
        query_layer_5 = view_15.transpose(1, 2)
        view_15 = None
        linear_31 = torch.nn.functional.linear(
            hidden_states_39,
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_16 = linear_31.view(1, s27, -1, 64)
        linear_31 = None
        key_layer_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_32 = torch.nn.functional.linear(
            hidden_states_39,
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_17 = linear_32.view(1, s27, -1, 64)
        linear_32 = None
        value_layer_5 = view_17.transpose(1, 2)
        view_17 = None
        item_30 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_scaling.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_scaling = (
            None
        )
        attn_output_15 = torch.nn.functional.scaled_dot_product_attention(
            query_layer_5,
            key_layer_5,
            value_layer_5,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=item_30,
            is_causal=False,
        )
        query_layer_5 = key_layer_5 = value_layer_5 = item_30 = None
        transpose_23 = attn_output_15.transpose(1, 2)
        attn_output_15 = None
        attn_output_16 = transpose_23.contiguous()
        transpose_23 = None
        reshape_5 = attn_output_16.reshape(1, s27, -1)
        attn_output_16 = None
        attn_output_17 = reshape_5.contiguous()
        reshape_5 = None
        hidden_states_40 = torch.nn.functional.linear(
            attn_output_17,
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_17 = l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_31 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_41 = torch.nn.functional.dropout(
            hidden_states_40, item_31, False, False
        )
        hidden_states_40 = item_31 = None
        add_16 = hidden_states_41 + hidden_states_39
        hidden_states_41 = hidden_states_39 = None
        item_32 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_42 = torch.nn.functional.layer_norm(
            add_16,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            item_32,
        )
        add_16 = l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (item_32) = (
            None
        )
        hidden_states_43 = torch.nn.functional.linear(
            hidden_states_42,
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_44 = torch.nn.functional.gelu(hidden_states_43)
        hidden_states_43 = None
        hidden_states_45 = torch.nn.functional.linear(
            hidden_states_44,
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_44 = l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = (None)
        item_33 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_46 = torch.nn.functional.dropout(
            hidden_states_45, item_33, False, False
        )
        hidden_states_45 = item_33 = None
        add_17 = hidden_states_46 + hidden_states_42
        hidden_states_46 = hidden_states_42 = None
        item_34 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_47 = torch.nn.functional.layer_norm(
            add_17,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_bias_,
            item_34,
        )
        add_17 = l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_5_modules_output_modules_layer_norm_parameters_bias_ = (item_34) = (
            None
        )
        linear_36 = torch.nn.functional.linear(
            hidden_states_47,
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_18 = linear_36.view(1, s27, -1, 64)
        linear_36 = None
        query_layer_6 = view_18.transpose(1, 2)
        view_18 = None
        linear_37 = torch.nn.functional.linear(
            hidden_states_47,
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_19 = linear_37.view(1, s27, -1, 64)
        linear_37 = None
        key_layer_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_38 = torch.nn.functional.linear(
            hidden_states_47,
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_20 = linear_38.view(1, s27, -1, 64)
        linear_38 = None
        value_layer_6 = view_20.transpose(1, 2)
        view_20 = None
        item_35 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_scaling.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_scaling = (
            None
        )
        attn_output_18 = torch.nn.functional.scaled_dot_product_attention(
            query_layer_6,
            key_layer_6,
            value_layer_6,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=item_35,
            is_causal=False,
        )
        query_layer_6 = key_layer_6 = value_layer_6 = item_35 = None
        transpose_27 = attn_output_18.transpose(1, 2)
        attn_output_18 = None
        attn_output_19 = transpose_27.contiguous()
        transpose_27 = None
        reshape_6 = attn_output_19.reshape(1, s27, -1)
        attn_output_19 = None
        attn_output_20 = reshape_6.contiguous()
        reshape_6 = None
        hidden_states_48 = torch.nn.functional.linear(
            attn_output_20,
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_20 = l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_36 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_49 = torch.nn.functional.dropout(
            hidden_states_48, item_36, False, False
        )
        hidden_states_48 = item_36 = None
        add_18 = hidden_states_49 + hidden_states_47
        hidden_states_49 = hidden_states_47 = None
        item_37 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_50 = torch.nn.functional.layer_norm(
            add_18,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            item_37,
        )
        add_18 = l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (item_37) = (
            None
        )
        hidden_states_51 = torch.nn.functional.linear(
            hidden_states_50,
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_52 = torch.nn.functional.gelu(hidden_states_51)
        hidden_states_51 = None
        hidden_states_53 = torch.nn.functional.linear(
            hidden_states_52,
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_52 = l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = (None)
        item_38 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_54 = torch.nn.functional.dropout(
            hidden_states_53, item_38, False, False
        )
        hidden_states_53 = item_38 = None
        add_19 = hidden_states_54 + hidden_states_50
        hidden_states_54 = hidden_states_50 = None
        item_39 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_55 = torch.nn.functional.layer_norm(
            add_19,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_bias_,
            item_39,
        )
        add_19 = l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_6_modules_output_modules_layer_norm_parameters_bias_ = (item_39) = (
            None
        )
        linear_42 = torch.nn.functional.linear(
            hidden_states_55,
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_21 = linear_42.view(1, s27, -1, 64)
        linear_42 = None
        query_layer_7 = view_21.transpose(1, 2)
        view_21 = None
        linear_43 = torch.nn.functional.linear(
            hidden_states_55,
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_22 = linear_43.view(1, s27, -1, 64)
        linear_43 = None
        key_layer_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_44 = torch.nn.functional.linear(
            hidden_states_55,
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_23 = linear_44.view(1, s27, -1, 64)
        linear_44 = None
        value_layer_7 = view_23.transpose(1, 2)
        view_23 = None
        item_40 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_scaling.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_scaling = (
            None
        )
        attn_output_21 = torch.nn.functional.scaled_dot_product_attention(
            query_layer_7,
            key_layer_7,
            value_layer_7,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=item_40,
            is_causal=False,
        )
        query_layer_7 = key_layer_7 = value_layer_7 = item_40 = None
        transpose_31 = attn_output_21.transpose(1, 2)
        attn_output_21 = None
        attn_output_22 = transpose_31.contiguous()
        transpose_31 = None
        reshape_7 = attn_output_22.reshape(1, s27, -1)
        attn_output_22 = None
        attn_output_23 = reshape_7.contiguous()
        reshape_7 = None
        hidden_states_56 = torch.nn.functional.linear(
            attn_output_23,
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_23 = l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_41 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_57 = torch.nn.functional.dropout(
            hidden_states_56, item_41, False, False
        )
        hidden_states_56 = item_41 = None
        add_20 = hidden_states_57 + hidden_states_55
        hidden_states_57 = hidden_states_55 = None
        item_42 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_58 = torch.nn.functional.layer_norm(
            add_20,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            item_42,
        )
        add_20 = l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (item_42) = (
            None
        )
        hidden_states_59 = torch.nn.functional.linear(
            hidden_states_58,
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_60 = torch.nn.functional.gelu(hidden_states_59)
        hidden_states_59 = None
        hidden_states_61 = torch.nn.functional.linear(
            hidden_states_60,
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_60 = l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = (None)
        item_43 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_62 = torch.nn.functional.dropout(
            hidden_states_61, item_43, False, False
        )
        hidden_states_61 = item_43 = None
        add_21 = hidden_states_62 + hidden_states_58
        hidden_states_62 = hidden_states_58 = None
        item_44 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_63 = torch.nn.functional.layer_norm(
            add_21,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_bias_,
            item_44,
        )
        add_21 = l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_7_modules_output_modules_layer_norm_parameters_bias_ = (item_44) = (
            None
        )
        linear_48 = torch.nn.functional.linear(
            hidden_states_63,
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_24 = linear_48.view(1, s27, -1, 64)
        linear_48 = None
        query_layer_8 = view_24.transpose(1, 2)
        view_24 = None
        linear_49 = torch.nn.functional.linear(
            hidden_states_63,
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_25 = linear_49.view(1, s27, -1, 64)
        linear_49 = None
        key_layer_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_50 = torch.nn.functional.linear(
            hidden_states_63,
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_26 = linear_50.view(1, s27, -1, 64)
        linear_50 = None
        value_layer_8 = view_26.transpose(1, 2)
        view_26 = None
        item_45 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_scaling.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_scaling = (
            None
        )
        attn_output_24 = torch.nn.functional.scaled_dot_product_attention(
            query_layer_8,
            key_layer_8,
            value_layer_8,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=item_45,
            is_causal=False,
        )
        query_layer_8 = key_layer_8 = value_layer_8 = item_45 = None
        transpose_35 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_35.contiguous()
        transpose_35 = None
        reshape_8 = attn_output_25.reshape(1, s27, -1)
        attn_output_25 = None
        attn_output_26 = reshape_8.contiguous()
        reshape_8 = None
        hidden_states_64 = torch.nn.functional.linear(
            attn_output_26,
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_26 = l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_46 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_65 = torch.nn.functional.dropout(
            hidden_states_64, item_46, False, False
        )
        hidden_states_64 = item_46 = None
        add_22 = hidden_states_65 + hidden_states_63
        hidden_states_65 = hidden_states_63 = None
        item_47 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_66 = torch.nn.functional.layer_norm(
            add_22,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            item_47,
        )
        add_22 = l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (item_47) = (
            None
        )
        hidden_states_67 = torch.nn.functional.linear(
            hidden_states_66,
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_68 = torch.nn.functional.gelu(hidden_states_67)
        hidden_states_67 = None
        hidden_states_69 = torch.nn.functional.linear(
            hidden_states_68,
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_68 = l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = (None)
        item_48 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, item_48, False, False
        )
        hidden_states_69 = item_48 = None
        add_23 = hidden_states_70 + hidden_states_66
        hidden_states_70 = hidden_states_66 = None
        item_49 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_71 = torch.nn.functional.layer_norm(
            add_23,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_bias_,
            item_49,
        )
        add_23 = l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_8_modules_output_modules_layer_norm_parameters_bias_ = (item_49) = (
            None
        )
        linear_54 = torch.nn.functional.linear(
            hidden_states_71,
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_27 = linear_54.view(1, s27, -1, 64)
        linear_54 = None
        query_layer_9 = view_27.transpose(1, 2)
        view_27 = None
        linear_55 = torch.nn.functional.linear(
            hidden_states_71,
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_28 = linear_55.view(1, s27, -1, 64)
        linear_55 = None
        key_layer_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_56 = torch.nn.functional.linear(
            hidden_states_71,
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_29 = linear_56.view(1, s27, -1, 64)
        linear_56 = None
        value_layer_9 = view_29.transpose(1, 2)
        view_29 = None
        item_50 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_scaling.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_scaling = (
            None
        )
        attn_output_27 = torch.nn.functional.scaled_dot_product_attention(
            query_layer_9,
            key_layer_9,
            value_layer_9,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=item_50,
            is_causal=False,
        )
        query_layer_9 = key_layer_9 = value_layer_9 = item_50 = None
        transpose_39 = attn_output_27.transpose(1, 2)
        attn_output_27 = None
        attn_output_28 = transpose_39.contiguous()
        transpose_39 = None
        reshape_9 = attn_output_28.reshape(1, s27, -1)
        attn_output_28 = None
        attn_output_29 = reshape_9.contiguous()
        reshape_9 = None
        hidden_states_72 = torch.nn.functional.linear(
            attn_output_29,
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_29 = l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_51 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_73 = torch.nn.functional.dropout(
            hidden_states_72, item_51, False, False
        )
        hidden_states_72 = item_51 = None
        add_24 = hidden_states_73 + hidden_states_71
        hidden_states_73 = hidden_states_71 = None
        item_52 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_74 = torch.nn.functional.layer_norm(
            add_24,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            item_52,
        )
        add_24 = l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (item_52) = (
            None
        )
        hidden_states_75 = torch.nn.functional.linear(
            hidden_states_74,
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_76 = torch.nn.functional.gelu(hidden_states_75)
        hidden_states_75 = None
        hidden_states_77 = torch.nn.functional.linear(
            hidden_states_76,
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_76 = l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = (None)
        item_53 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_78 = torch.nn.functional.dropout(
            hidden_states_77, item_53, False, False
        )
        hidden_states_77 = item_53 = None
        add_25 = hidden_states_78 + hidden_states_74
        hidden_states_78 = hidden_states_74 = None
        item_54 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_79 = torch.nn.functional.layer_norm(
            add_25,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_bias_,
            item_54,
        )
        add_25 = l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_9_modules_output_modules_layer_norm_parameters_bias_ = (item_54) = (
            None
        )
        linear_60 = torch.nn.functional.linear(
            hidden_states_79,
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_30 = linear_60.view(1, s27, -1, 64)
        linear_60 = None
        query_layer_10 = view_30.transpose(1, 2)
        view_30 = None
        linear_61 = torch.nn.functional.linear(
            hidden_states_79,
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_31 = linear_61.view(1, s27, -1, 64)
        linear_61 = None
        key_layer_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_62 = torch.nn.functional.linear(
            hidden_states_79,
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_32 = linear_62.view(1, s27, -1, 64)
        linear_62 = None
        value_layer_10 = view_32.transpose(1, 2)
        view_32 = None
        item_55 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_scaling.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_scaling = (
            None
        )
        attn_output_30 = torch.nn.functional.scaled_dot_product_attention(
            query_layer_10,
            key_layer_10,
            value_layer_10,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=item_55,
            is_causal=False,
        )
        query_layer_10 = key_layer_10 = value_layer_10 = item_55 = None
        transpose_43 = attn_output_30.transpose(1, 2)
        attn_output_30 = None
        attn_output_31 = transpose_43.contiguous()
        transpose_43 = None
        reshape_10 = attn_output_31.reshape(1, s27, -1)
        attn_output_31 = None
        attn_output_32 = reshape_10.contiguous()
        reshape_10 = None
        hidden_states_80 = torch.nn.functional.linear(
            attn_output_32,
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_32 = l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_56 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_81 = torch.nn.functional.dropout(
            hidden_states_80, item_56, False, False
        )
        hidden_states_80 = item_56 = None
        add_26 = hidden_states_81 + hidden_states_79
        hidden_states_81 = hidden_states_79 = None
        item_57 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_82 = torch.nn.functional.layer_norm(
            add_26,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            item_57,
        )
        add_26 = l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (item_57) = (
            None
        )
        hidden_states_83 = torch.nn.functional.linear(
            hidden_states_82,
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_84 = torch.nn.functional.gelu(hidden_states_83)
        hidden_states_83 = None
        hidden_states_85 = torch.nn.functional.linear(
            hidden_states_84,
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_84 = l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = (None)
        item_58 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_86 = torch.nn.functional.dropout(
            hidden_states_85, item_58, False, False
        )
        hidden_states_85 = item_58 = None
        add_27 = hidden_states_86 + hidden_states_82
        hidden_states_86 = hidden_states_82 = None
        item_59 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_87 = torch.nn.functional.layer_norm(
            add_27,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_bias_,
            item_59,
        )
        add_27 = l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_10_modules_output_modules_layer_norm_parameters_bias_ = (item_59) = (
            None
        )
        linear_66 = torch.nn.functional.linear(
            hidden_states_87,
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_33 = linear_66.view(1, s27, -1, 64)
        linear_66 = None
        query_layer_11 = view_33.transpose(1, 2)
        view_33 = None
        linear_67 = torch.nn.functional.linear(
            hidden_states_87,
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_34 = linear_67.view(1, s27, -1, 64)
        linear_67 = None
        key_layer_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_68 = torch.nn.functional.linear(
            hidden_states_87,
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_35 = linear_68.view(1, s27, -1, 64)
        linear_68 = None
        value_layer_11 = view_35.transpose(1, 2)
        view_35 = None
        item_60 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_scaling.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_scaling = (
            None
        )
        attn_output_33 = torch.nn.functional.scaled_dot_product_attention(
            query_layer_11,
            key_layer_11,
            value_layer_11,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=item_60,
            is_causal=False,
        )
        query_layer_11 = (
            key_layer_11
        ) = value_layer_11 = attention_mask_1 = item_60 = None
        transpose_47 = attn_output_33.transpose(1, 2)
        attn_output_33 = None
        attn_output_34 = transpose_47.contiguous()
        transpose_47 = None
        reshape_11 = attn_output_34.reshape(1, s27, -1)
        attn_output_34 = s27 = None
        attn_output_35 = reshape_11.contiguous()
        reshape_11 = None
        hidden_states_88 = torch.nn.functional.linear(
            attn_output_35,
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        attn_output_35 = l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_61 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_89 = torch.nn.functional.dropout(
            hidden_states_88, item_61, False, False
        )
        hidden_states_88 = item_61 = None
        add_28 = hidden_states_89 + hidden_states_87
        hidden_states_89 = hidden_states_87 = None
        item_62 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_90 = torch.nn.functional.layer_norm(
            add_28,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_bias_,
            item_62,
        )
        add_28 = l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_layer_norm_parameters_bias_ = (item_62) = (
            None
        )
        hidden_states_91 = torch.nn.functional.linear(
            hidden_states_90,
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_,
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_92 = torch.nn.functional.gelu(hidden_states_91)
        hidden_states_91 = None
        hidden_states_93 = torch.nn.functional.linear(
            hidden_states_92,
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = (None)
        item_63 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_94 = torch.nn.functional.dropout(
            hidden_states_93, item_63, False, False
        )
        hidden_states_93 = item_63 = None
        add_29 = hidden_states_94 + hidden_states_90
        hidden_states_94 = hidden_states_90 = None
        item_64 = (
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_eps.item()
        )
        l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_eps = (
            None
        )
        hidden_states_95 = torch.nn.functional.layer_norm(
            add_29,
            (768,),
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_weight_,
            l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_bias_,
            item_64,
        )
        add_29 = l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_weight_ = l_self_modules_bert_modules_encoder_modules_layer_modules_11_modules_output_modules_layer_norm_parameters_bias_ = (item_64) = (
            None
        )
        item_65 = l_self_modules_dropout_p.item()
        l_self_modules_dropout_p = None
        sequence_output = torch.nn.functional.dropout(
            hidden_states_95, item_65, False, False
        )
        hidden_states_95 = item_65 = None
        logits = torch.nn.functional.linear(
            sequence_output,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        sequence_output = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (logits,)
