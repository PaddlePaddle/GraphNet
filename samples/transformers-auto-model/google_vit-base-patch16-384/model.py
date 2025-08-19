import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_pixel_values_: torch.Tensor,
        L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_parameters_position_embeddings_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layernorm_eps: torch.Tensor,
        L_self_modules_pooler_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_pixel_values_ = L_pixel_values_
        l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_ = L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_
        l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_ = L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_
        l_self_modules_embeddings_parameters_cls_token_ = (
            L_self_modules_embeddings_parameters_cls_token_
        )
        l_self_modules_embeddings_parameters_position_embeddings_ = (
            L_self_modules_embeddings_parameters_position_embeddings_
        )
        l_self_modules_embeddings_modules_dropout_p = (
            L_self_modules_embeddings_modules_dropout_p
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p
        l_self_modules_layernorm_parameters_weight_ = (
            L_self_modules_layernorm_parameters_weight_
        )
        l_self_modules_layernorm_parameters_bias_ = (
            L_self_modules_layernorm_parameters_bias_
        )
        l_self_modules_layernorm_eps = L_self_modules_layernorm_eps
        l_self_modules_pooler_modules_dense_parameters_weight_ = (
            L_self_modules_pooler_modules_dense_parameters_weight_
        )
        l_self_modules_pooler_modules_dense_parameters_bias_ = (
            L_self_modules_pooler_modules_dense_parameters_bias_
        )
        conv2d = torch.conv2d(
            l_pixel_values_,
            l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_,
            l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        l_pixel_values_ = l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_ = l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_ = (None)
        flatten = conv2d.flatten(2)
        conv2d = None
        embeddings = flatten.transpose(1, 2)
        flatten = None
        cls_tokens = l_self_modules_embeddings_parameters_cls_token_.expand(1, -1, -1)
        l_self_modules_embeddings_parameters_cls_token_ = None
        embeddings_1 = torch.cat((cls_tokens, embeddings), dim=1)
        cls_tokens = embeddings = None
        embeddings_2 = (
            embeddings_1 + l_self_modules_embeddings_parameters_position_embeddings_
        )
        embeddings_1 = l_self_modules_embeddings_parameters_position_embeddings_ = None
        item = l_self_modules_embeddings_modules_dropout_p.item()
        l_self_modules_embeddings_modules_dropout_p = None
        embeddings_3 = torch.nn.functional.dropout(embeddings_2, item, False, False)
        embeddings_2 = item = None
        item_1 = (
            l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_eps = (
            None
        )
        layer_norm = torch.nn.functional.layer_norm(
            embeddings_3,
            (768,),
            l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_,
            item_1,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_ = (item_1) = (
            None
        )
        linear = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view = linear.view(1, -1, 12, 64)
        linear = None
        key_layer = view.transpose(1, 2)
        view = None
        linear_1 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_1 = linear_1.view(1, -1, 12, 64)
        linear_1 = None
        value_layer = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_2 = linear_2.view(1, -1, 12, 64)
        linear_2 = None
        query_layer = view_2.transpose(1, 2)
        view_2 = None
        query = query_layer.contiguous()
        query_layer = None
        key = key_layer.contiguous()
        key_layer = None
        value = value_layer.contiguous()
        value_layer = None
        item_2 = (
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_2,
            is_causal=False,
        )
        query = key = value = item_2 = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        context_layer = attn_output_1.reshape((1, 577, 768))
        attn_output_1 = None
        hidden_states = torch._C._nn.linear(
            context_layer,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_3 = (
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_1 = torch.nn.functional.dropout(
            hidden_states, item_3, False, False
        )
        hidden_states = item_3 = None
        hidden_states_2 = hidden_states_1 + embeddings_3
        hidden_states_1 = embeddings_3 = None
        item_4 = (
            l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_eps = (
            None
        )
        layer_output = torch.nn.functional.layer_norm(
            hidden_states_2,
            (768,),
            l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_,
            item_4,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_ = (item_4) = (
            None
        )
        hidden_states_3 = torch._C._nn.linear(
            layer_output,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output = l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_4 = torch._C._nn.gelu(hidden_states_3)
        hidden_states_3 = None
        hidden_states_5 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_4 = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        item_5 = (
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, item_5, False, False
        )
        hidden_states_5 = item_5 = None
        hidden_states_7 = hidden_states_6 + hidden_states_2
        hidden_states_6 = hidden_states_2 = None
        item_6 = (
            l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_eps = (
            None
        )
        layer_norm_2 = torch.nn.functional.layer_norm(
            hidden_states_7,
            (768,),
            l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_,
            item_6,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_ = (item_6) = (
            None
        )
        linear_6 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_3 = linear_6.view(1, -1, 12, 64)
        linear_6 = None
        key_layer_1 = view_3.transpose(1, 2)
        view_3 = None
        linear_7 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_4 = linear_7.view(1, -1, 12, 64)
        linear_7 = None
        value_layer_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_8 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_2 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_5 = linear_8.view(1, -1, 12, 64)
        linear_8 = None
        query_layer_1 = view_5.transpose(1, 2)
        view_5 = None
        query_1 = query_layer_1.contiguous()
        query_layer_1 = None
        key_1 = key_layer_1.contiguous()
        key_layer_1 = None
        value_1 = value_layer_1.contiguous()
        value_layer_1 = None
        item_7 = (
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_2 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_7,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = item_7 = None
        transpose_8 = attn_output_2.transpose(1, 2)
        attn_output_2 = None
        attn_output_3 = transpose_8.contiguous()
        transpose_8 = None
        context_layer_1 = attn_output_3.reshape((1, 577, 768))
        attn_output_3 = None
        hidden_states_8 = torch._C._nn.linear(
            context_layer_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_1 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_8 = (
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, item_8, False, False
        )
        hidden_states_8 = item_8 = None
        hidden_states_10 = hidden_states_9 + hidden_states_7
        hidden_states_9 = hidden_states_7 = None
        item_9 = (
            l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_eps = (
            None
        )
        layer_output_1 = torch.nn.functional.layer_norm(
            hidden_states_10,
            (768,),
            l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_,
            item_9,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_ = (item_9) = (
            None
        )
        hidden_states_11 = torch._C._nn.linear(
            layer_output_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_1 = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_12 = torch._C._nn.gelu(hidden_states_11)
        hidden_states_11 = None
        hidden_states_13 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_12 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        item_10 = (
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_14 = torch.nn.functional.dropout(
            hidden_states_13, item_10, False, False
        )
        hidden_states_13 = item_10 = None
        hidden_states_15 = hidden_states_14 + hidden_states_10
        hidden_states_14 = hidden_states_10 = None
        item_11 = (
            l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_eps = (
            None
        )
        layer_norm_4 = torch.nn.functional.layer_norm(
            hidden_states_15,
            (768,),
            l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_,
            item_11,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_ = (item_11) = (
            None
        )
        linear_12 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_6 = linear_12.view(1, -1, 12, 64)
        linear_12 = None
        key_layer_2 = view_6.transpose(1, 2)
        view_6 = None
        linear_13 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_7 = linear_13.view(1, -1, 12, 64)
        linear_13 = None
        value_layer_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_14 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_4 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_8 = linear_14.view(1, -1, 12, 64)
        linear_14 = None
        query_layer_2 = view_8.transpose(1, 2)
        view_8 = None
        query_2 = query_layer_2.contiguous()
        query_layer_2 = None
        key_2 = key_layer_2.contiguous()
        key_layer_2 = None
        value_2 = value_layer_2.contiguous()
        value_layer_2 = None
        item_12 = (
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_12,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = item_12 = None
        transpose_12 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_12.contiguous()
        transpose_12 = None
        context_layer_2 = attn_output_5.reshape((1, 577, 768))
        attn_output_5 = None
        hidden_states_16 = torch._C._nn.linear(
            context_layer_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_13 = (
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_17 = torch.nn.functional.dropout(
            hidden_states_16, item_13, False, False
        )
        hidden_states_16 = item_13 = None
        hidden_states_18 = hidden_states_17 + hidden_states_15
        hidden_states_17 = hidden_states_15 = None
        item_14 = (
            l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_eps = (
            None
        )
        layer_output_2 = torch.nn.functional.layer_norm(
            hidden_states_18,
            (768,),
            l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_,
            item_14,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_ = (item_14) = (
            None
        )
        hidden_states_19 = torch._C._nn.linear(
            layer_output_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_2 = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_20 = torch._C._nn.gelu(hidden_states_19)
        hidden_states_19 = None
        hidden_states_21 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        item_15 = (
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, item_15, False, False
        )
        hidden_states_21 = item_15 = None
        hidden_states_23 = hidden_states_22 + hidden_states_18
        hidden_states_22 = hidden_states_18 = None
        item_16 = (
            l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_eps = (
            None
        )
        layer_norm_6 = torch.nn.functional.layer_norm(
            hidden_states_23,
            (768,),
            l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_,
            item_16,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_ = (item_16) = (
            None
        )
        linear_18 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_9 = linear_18.view(1, -1, 12, 64)
        linear_18 = None
        key_layer_3 = view_9.transpose(1, 2)
        view_9 = None
        linear_19 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_10 = linear_19.view(1, -1, 12, 64)
        linear_19 = None
        value_layer_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_20 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_6 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_11 = linear_20.view(1, -1, 12, 64)
        linear_20 = None
        query_layer_3 = view_11.transpose(1, 2)
        view_11 = None
        query_3 = query_layer_3.contiguous()
        query_layer_3 = None
        key_3 = key_layer_3.contiguous()
        key_layer_3 = None
        value_3 = value_layer_3.contiguous()
        value_layer_3 = None
        item_17 = (
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_6 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_17,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = item_17 = None
        transpose_16 = attn_output_6.transpose(1, 2)
        attn_output_6 = None
        attn_output_7 = transpose_16.contiguous()
        transpose_16 = None
        context_layer_3 = attn_output_7.reshape((1, 577, 768))
        attn_output_7 = None
        hidden_states_24 = torch._C._nn.linear(
            context_layer_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_3 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_18 = (
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, item_18, False, False
        )
        hidden_states_24 = item_18 = None
        hidden_states_26 = hidden_states_25 + hidden_states_23
        hidden_states_25 = hidden_states_23 = None
        item_19 = (
            l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_eps = (
            None
        )
        layer_output_3 = torch.nn.functional.layer_norm(
            hidden_states_26,
            (768,),
            l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_,
            item_19,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_ = (item_19) = (
            None
        )
        hidden_states_27 = torch._C._nn.linear(
            layer_output_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_3 = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_28 = torch._C._nn.gelu(hidden_states_27)
        hidden_states_27 = None
        hidden_states_29 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_28 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = (None)
        item_20 = (
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_30 = torch.nn.functional.dropout(
            hidden_states_29, item_20, False, False
        )
        hidden_states_29 = item_20 = None
        hidden_states_31 = hidden_states_30 + hidden_states_26
        hidden_states_30 = hidden_states_26 = None
        item_21 = (
            l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_eps = (
            None
        )
        layer_norm_8 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (768,),
            l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_,
            item_21,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_ = (item_21) = (
            None
        )
        linear_24 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_12 = linear_24.view(1, -1, 12, 64)
        linear_24 = None
        key_layer_4 = view_12.transpose(1, 2)
        view_12 = None
        linear_25 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_13 = linear_25.view(1, -1, 12, 64)
        linear_25 = None
        value_layer_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_26 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_8 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_14 = linear_26.view(1, -1, 12, 64)
        linear_26 = None
        query_layer_4 = view_14.transpose(1, 2)
        view_14 = None
        query_4 = query_layer_4.contiguous()
        query_layer_4 = None
        key_4 = key_layer_4.contiguous()
        key_layer_4 = None
        value_4 = value_layer_4.contiguous()
        value_layer_4 = None
        item_22 = (
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_22,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = item_22 = None
        transpose_20 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_20.contiguous()
        transpose_20 = None
        context_layer_4 = attn_output_9.reshape((1, 577, 768))
        attn_output_9 = None
        hidden_states_32 = torch._C._nn.linear(
            context_layer_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_4 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_23 = (
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_33 = torch.nn.functional.dropout(
            hidden_states_32, item_23, False, False
        )
        hidden_states_32 = item_23 = None
        hidden_states_34 = hidden_states_33 + hidden_states_31
        hidden_states_33 = hidden_states_31 = None
        item_24 = (
            l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_eps = (
            None
        )
        layer_output_4 = torch.nn.functional.layer_norm(
            hidden_states_34,
            (768,),
            l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_,
            item_24,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_ = (item_24) = (
            None
        )
        hidden_states_35 = torch._C._nn.linear(
            layer_output_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_4 = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_36 = torch._C._nn.gelu(hidden_states_35)
        hidden_states_35 = None
        hidden_states_37 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_36 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = (None)
        item_25 = (
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_38 = torch.nn.functional.dropout(
            hidden_states_37, item_25, False, False
        )
        hidden_states_37 = item_25 = None
        hidden_states_39 = hidden_states_38 + hidden_states_34
        hidden_states_38 = hidden_states_34 = None
        item_26 = (
            l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_eps = (
            None
        )
        layer_norm_10 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (768,),
            l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_,
            item_26,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_ = (item_26) = (
            None
        )
        linear_30 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_15 = linear_30.view(1, -1, 12, 64)
        linear_30 = None
        key_layer_5 = view_15.transpose(1, 2)
        view_15 = None
        linear_31 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_16 = linear_31.view(1, -1, 12, 64)
        linear_31 = None
        value_layer_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_32 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_10 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_17 = linear_32.view(1, -1, 12, 64)
        linear_32 = None
        query_layer_5 = view_17.transpose(1, 2)
        view_17 = None
        query_5 = query_layer_5.contiguous()
        query_layer_5 = None
        key_5 = key_layer_5.contiguous()
        key_layer_5 = None
        value_5 = value_layer_5.contiguous()
        value_layer_5 = None
        item_27 = (
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_10 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_27,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = item_27 = None
        transpose_24 = attn_output_10.transpose(1, 2)
        attn_output_10 = None
        attn_output_11 = transpose_24.contiguous()
        transpose_24 = None
        context_layer_5 = attn_output_11.reshape((1, 577, 768))
        attn_output_11 = None
        hidden_states_40 = torch._C._nn.linear(
            context_layer_5,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_5 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_28 = (
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_41 = torch.nn.functional.dropout(
            hidden_states_40, item_28, False, False
        )
        hidden_states_40 = item_28 = None
        hidden_states_42 = hidden_states_41 + hidden_states_39
        hidden_states_41 = hidden_states_39 = None
        item_29 = (
            l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_eps = (
            None
        )
        layer_output_5 = torch.nn.functional.layer_norm(
            hidden_states_42,
            (768,),
            l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_,
            item_29,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_ = (item_29) = (
            None
        )
        hidden_states_43 = torch._C._nn.linear(
            layer_output_5,
            l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_5 = l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_44 = torch._C._nn.gelu(hidden_states_43)
        hidden_states_43 = None
        hidden_states_45 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_44 = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = (None)
        item_30 = (
            l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_46 = torch.nn.functional.dropout(
            hidden_states_45, item_30, False, False
        )
        hidden_states_45 = item_30 = None
        hidden_states_47 = hidden_states_46 + hidden_states_42
        hidden_states_46 = hidden_states_42 = None
        item_31 = (
            l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_eps = (
            None
        )
        layer_norm_12 = torch.nn.functional.layer_norm(
            hidden_states_47,
            (768,),
            l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_,
            item_31,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_ = (item_31) = (
            None
        )
        linear_36 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_18 = linear_36.view(1, -1, 12, 64)
        linear_36 = None
        key_layer_6 = view_18.transpose(1, 2)
        view_18 = None
        linear_37 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_19 = linear_37.view(1, -1, 12, 64)
        linear_37 = None
        value_layer_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_38 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_12 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_20 = linear_38.view(1, -1, 12, 64)
        linear_38 = None
        query_layer_6 = view_20.transpose(1, 2)
        view_20 = None
        query_6 = query_layer_6.contiguous()
        query_layer_6 = None
        key_6 = key_layer_6.contiguous()
        key_layer_6 = None
        value_6 = value_layer_6.contiguous()
        value_layer_6 = None
        item_32 = (
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_6,
            value_6,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_32,
            is_causal=False,
        )
        query_6 = key_6 = value_6 = item_32 = None
        transpose_28 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_28.contiguous()
        transpose_28 = None
        context_layer_6 = attn_output_13.reshape((1, 577, 768))
        attn_output_13 = None
        hidden_states_48 = torch._C._nn.linear(
            context_layer_6,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_6 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_33 = (
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_49 = torch.nn.functional.dropout(
            hidden_states_48, item_33, False, False
        )
        hidden_states_48 = item_33 = None
        hidden_states_50 = hidden_states_49 + hidden_states_47
        hidden_states_49 = hidden_states_47 = None
        item_34 = (
            l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_eps = (
            None
        )
        layer_output_6 = torch.nn.functional.layer_norm(
            hidden_states_50,
            (768,),
            l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_,
            item_34,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_ = (item_34) = (
            None
        )
        hidden_states_51 = torch._C._nn.linear(
            layer_output_6,
            l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_6 = l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_52 = torch._C._nn.gelu(hidden_states_51)
        hidden_states_51 = None
        hidden_states_53 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_52 = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = (None)
        item_35 = (
            l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_54 = torch.nn.functional.dropout(
            hidden_states_53, item_35, False, False
        )
        hidden_states_53 = item_35 = None
        hidden_states_55 = hidden_states_54 + hidden_states_50
        hidden_states_54 = hidden_states_50 = None
        item_36 = (
            l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_eps = (
            None
        )
        layer_norm_14 = torch.nn.functional.layer_norm(
            hidden_states_55,
            (768,),
            l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_,
            item_36,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_ = (item_36) = (
            None
        )
        linear_42 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_21 = linear_42.view(1, -1, 12, 64)
        linear_42 = None
        key_layer_7 = view_21.transpose(1, 2)
        view_21 = None
        linear_43 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_22 = linear_43.view(1, -1, 12, 64)
        linear_43 = None
        value_layer_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_44 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_14 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_23 = linear_44.view(1, -1, 12, 64)
        linear_44 = None
        query_layer_7 = view_23.transpose(1, 2)
        view_23 = None
        query_7 = query_layer_7.contiguous()
        query_layer_7 = None
        key_7 = key_layer_7.contiguous()
        key_layer_7 = None
        value_7 = value_layer_7.contiguous()
        value_layer_7 = None
        item_37 = (
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_14 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_7,
            value_7,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_37,
            is_causal=False,
        )
        query_7 = key_7 = value_7 = item_37 = None
        transpose_32 = attn_output_14.transpose(1, 2)
        attn_output_14 = None
        attn_output_15 = transpose_32.contiguous()
        transpose_32 = None
        context_layer_7 = attn_output_15.reshape((1, 577, 768))
        attn_output_15 = None
        hidden_states_56 = torch._C._nn.linear(
            context_layer_7,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_7 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_38 = (
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_57 = torch.nn.functional.dropout(
            hidden_states_56, item_38, False, False
        )
        hidden_states_56 = item_38 = None
        hidden_states_58 = hidden_states_57 + hidden_states_55
        hidden_states_57 = hidden_states_55 = None
        item_39 = (
            l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_eps = (
            None
        )
        layer_output_7 = torch.nn.functional.layer_norm(
            hidden_states_58,
            (768,),
            l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_,
            item_39,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_ = (item_39) = (
            None
        )
        hidden_states_59 = torch._C._nn.linear(
            layer_output_7,
            l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_7 = l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_60 = torch._C._nn.gelu(hidden_states_59)
        hidden_states_59 = None
        hidden_states_61 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_60 = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = (None)
        item_40 = (
            l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_62 = torch.nn.functional.dropout(
            hidden_states_61, item_40, False, False
        )
        hidden_states_61 = item_40 = None
        hidden_states_63 = hidden_states_62 + hidden_states_58
        hidden_states_62 = hidden_states_58 = None
        item_41 = (
            l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_eps = (
            None
        )
        layer_norm_16 = torch.nn.functional.layer_norm(
            hidden_states_63,
            (768,),
            l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_,
            item_41,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_ = (item_41) = (
            None
        )
        linear_48 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_24 = linear_48.view(1, -1, 12, 64)
        linear_48 = None
        key_layer_8 = view_24.transpose(1, 2)
        view_24 = None
        linear_49 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_25 = linear_49.view(1, -1, 12, 64)
        linear_49 = None
        value_layer_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_50 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_26 = linear_50.view(1, -1, 12, 64)
        linear_50 = None
        query_layer_8 = view_26.transpose(1, 2)
        view_26 = None
        query_8 = query_layer_8.contiguous()
        query_layer_8 = None
        key_8 = key_layer_8.contiguous()
        key_layer_8 = None
        value_8 = value_layer_8.contiguous()
        value_layer_8 = None
        item_42 = (
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_8,
            value_8,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_42,
            is_causal=False,
        )
        query_8 = key_8 = value_8 = item_42 = None
        transpose_36 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_36.contiguous()
        transpose_36 = None
        context_layer_8 = attn_output_17.reshape((1, 577, 768))
        attn_output_17 = None
        hidden_states_64 = torch._C._nn.linear(
            context_layer_8,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_8 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_43 = (
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_65 = torch.nn.functional.dropout(
            hidden_states_64, item_43, False, False
        )
        hidden_states_64 = item_43 = None
        hidden_states_66 = hidden_states_65 + hidden_states_63
        hidden_states_65 = hidden_states_63 = None
        item_44 = (
            l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_eps = (
            None
        )
        layer_output_8 = torch.nn.functional.layer_norm(
            hidden_states_66,
            (768,),
            l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_,
            item_44,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_ = (item_44) = (
            None
        )
        hidden_states_67 = torch._C._nn.linear(
            layer_output_8,
            l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_8 = l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_68 = torch._C._nn.gelu(hidden_states_67)
        hidden_states_67 = None
        hidden_states_69 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_68 = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = (None)
        item_45 = (
            l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, item_45, False, False
        )
        hidden_states_69 = item_45 = None
        hidden_states_71 = hidden_states_70 + hidden_states_66
        hidden_states_70 = hidden_states_66 = None
        item_46 = (
            l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_eps = (
            None
        )
        layer_norm_18 = torch.nn.functional.layer_norm(
            hidden_states_71,
            (768,),
            l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_,
            item_46,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_ = (item_46) = (
            None
        )
        linear_54 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_27 = linear_54.view(1, -1, 12, 64)
        linear_54 = None
        key_layer_9 = view_27.transpose(1, 2)
        view_27 = None
        linear_55 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_28 = linear_55.view(1, -1, 12, 64)
        linear_55 = None
        value_layer_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_56 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_18 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_29 = linear_56.view(1, -1, 12, 64)
        linear_56 = None
        query_layer_9 = view_29.transpose(1, 2)
        view_29 = None
        query_9 = query_layer_9.contiguous()
        query_layer_9 = None
        key_9 = key_layer_9.contiguous()
        key_layer_9 = None
        value_9 = value_layer_9.contiguous()
        value_layer_9 = None
        item_47 = (
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_18 = torch._C._nn.scaled_dot_product_attention(
            query_9,
            key_9,
            value_9,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_47,
            is_causal=False,
        )
        query_9 = key_9 = value_9 = item_47 = None
        transpose_40 = attn_output_18.transpose(1, 2)
        attn_output_18 = None
        attn_output_19 = transpose_40.contiguous()
        transpose_40 = None
        context_layer_9 = attn_output_19.reshape((1, 577, 768))
        attn_output_19 = None
        hidden_states_72 = torch._C._nn.linear(
            context_layer_9,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_9 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_48 = (
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_73 = torch.nn.functional.dropout(
            hidden_states_72, item_48, False, False
        )
        hidden_states_72 = item_48 = None
        hidden_states_74 = hidden_states_73 + hidden_states_71
        hidden_states_73 = hidden_states_71 = None
        item_49 = (
            l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_eps = (
            None
        )
        layer_output_9 = torch.nn.functional.layer_norm(
            hidden_states_74,
            (768,),
            l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_,
            item_49,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_ = (item_49) = (
            None
        )
        hidden_states_75 = torch._C._nn.linear(
            layer_output_9,
            l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_9 = l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_76 = torch._C._nn.gelu(hidden_states_75)
        hidden_states_75 = None
        hidden_states_77 = torch._C._nn.linear(
            hidden_states_76,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_76 = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = (None)
        item_50 = (
            l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_78 = torch.nn.functional.dropout(
            hidden_states_77, item_50, False, False
        )
        hidden_states_77 = item_50 = None
        hidden_states_79 = hidden_states_78 + hidden_states_74
        hidden_states_78 = hidden_states_74 = None
        item_51 = (
            l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_eps = (
            None
        )
        layer_norm_20 = torch.nn.functional.layer_norm(
            hidden_states_79,
            (768,),
            l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_,
            item_51,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_ = (item_51) = (
            None
        )
        linear_60 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_30 = linear_60.view(1, -1, 12, 64)
        linear_60 = None
        key_layer_10 = view_30.transpose(1, 2)
        view_30 = None
        linear_61 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_31 = linear_61.view(1, -1, 12, 64)
        linear_61 = None
        value_layer_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_62 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_20 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_32 = linear_62.view(1, -1, 12, 64)
        linear_62 = None
        query_layer_10 = view_32.transpose(1, 2)
        view_32 = None
        query_10 = query_layer_10.contiguous()
        query_layer_10 = None
        key_10 = key_layer_10.contiguous()
        key_layer_10 = None
        value_10 = value_layer_10.contiguous()
        value_layer_10 = None
        item_52 = (
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_10,
            key_10,
            value_10,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_52,
            is_causal=False,
        )
        query_10 = key_10 = value_10 = item_52 = None
        transpose_44 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_44.contiguous()
        transpose_44 = None
        context_layer_10 = attn_output_21.reshape((1, 577, 768))
        attn_output_21 = None
        hidden_states_80 = torch._C._nn.linear(
            context_layer_10,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_10 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_53 = (
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_81 = torch.nn.functional.dropout(
            hidden_states_80, item_53, False, False
        )
        hidden_states_80 = item_53 = None
        hidden_states_82 = hidden_states_81 + hidden_states_79
        hidden_states_81 = hidden_states_79 = None
        item_54 = (
            l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_eps = (
            None
        )
        layer_output_10 = torch.nn.functional.layer_norm(
            hidden_states_82,
            (768,),
            l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_,
            item_54,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_ = (item_54) = (
            None
        )
        hidden_states_83 = torch._C._nn.linear(
            layer_output_10,
            l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_10 = l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_84 = torch._C._nn.gelu(hidden_states_83)
        hidden_states_83 = None
        hidden_states_85 = torch._C._nn.linear(
            hidden_states_84,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_84 = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = (None)
        item_55 = (
            l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_86 = torch.nn.functional.dropout(
            hidden_states_85, item_55, False, False
        )
        hidden_states_85 = item_55 = None
        hidden_states_87 = hidden_states_86 + hidden_states_82
        hidden_states_86 = hidden_states_82 = None
        item_56 = (
            l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_eps = (
            None
        )
        layer_norm_22 = torch.nn.functional.layer_norm(
            hidden_states_87,
            (768,),
            l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_,
            item_56,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_ = (item_56) = (
            None
        )
        linear_66 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_33 = linear_66.view(1, -1, 12, 64)
        linear_66 = None
        key_layer_11 = view_33.transpose(1, 2)
        view_33 = None
        linear_67 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_34 = linear_67.view(1, -1, 12, 64)
        linear_67 = None
        value_layer_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_68 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_22 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_35 = linear_68.view(1, -1, 12, 64)
        linear_68 = None
        query_layer_11 = view_35.transpose(1, 2)
        view_35 = None
        query_11 = query_layer_11.contiguous()
        query_layer_11 = None
        key_11 = key_layer_11.contiguous()
        key_layer_11 = None
        value_11 = value_layer_11.contiguous()
        value_layer_11 = None
        item_57 = (
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_22 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_11,
            value_11,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_57,
            is_causal=False,
        )
        query_11 = key_11 = value_11 = item_57 = None
        transpose_48 = attn_output_22.transpose(1, 2)
        attn_output_22 = None
        attn_output_23 = transpose_48.contiguous()
        transpose_48 = None
        context_layer_11 = attn_output_23.reshape((1, 577, 768))
        attn_output_23 = None
        hidden_states_88 = torch._C._nn.linear(
            context_layer_11,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_58 = (
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_89 = torch.nn.functional.dropout(
            hidden_states_88, item_58, False, False
        )
        hidden_states_88 = item_58 = None
        hidden_states_90 = hidden_states_89 + hidden_states_87
        hidden_states_89 = hidden_states_87 = None
        item_59 = (
            l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_eps = (
            None
        )
        layer_output_11 = torch.nn.functional.layer_norm(
            hidden_states_90,
            (768,),
            l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_,
            item_59,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_ = (item_59) = (
            None
        )
        hidden_states_91 = torch._C._nn.linear(
            layer_output_11,
            l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_11 = l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_92 = torch._C._nn.gelu(hidden_states_91)
        hidden_states_91 = None
        hidden_states_93 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = (None)
        item_60 = (
            l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_94 = torch.nn.functional.dropout(
            hidden_states_93, item_60, False, False
        )
        hidden_states_93 = item_60 = None
        hidden_states_95 = hidden_states_94 + hidden_states_90
        hidden_states_94 = hidden_states_90 = None
        item_61 = l_self_modules_layernorm_eps.item()
        l_self_modules_layernorm_eps = None
        sequence_output = torch.nn.functional.layer_norm(
            hidden_states_95,
            (768,),
            l_self_modules_layernorm_parameters_weight_,
            l_self_modules_layernorm_parameters_bias_,
            item_61,
        )
        hidden_states_95 = (
            l_self_modules_layernorm_parameters_weight_
        ) = l_self_modules_layernorm_parameters_bias_ = item_61 = None
        first_token_tensor = sequence_output[(slice(None, None, None), 0)]
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
        return (sequence_output, pooled_output_1)
