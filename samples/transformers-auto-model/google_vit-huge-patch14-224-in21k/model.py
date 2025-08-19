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
        L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dropout_p: torch.Tensor,
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
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_eps = (
            L_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_eps
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_eps = (
            L_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_eps
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dropout_p
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
            (14, 14),
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
            (1280,),
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
        view = linear.view(1, -1, 16, 80)
        linear = None
        key_layer = view.transpose(1, 2)
        view = None
        linear_1 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_1 = linear_1.view(1, -1, 16, 80)
        linear_1 = None
        value_layer = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_2 = linear_2.view(1, -1, 16, 80)
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
        context_layer = attn_output_1.reshape((1, 257, 1280))
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
            (1280,),
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
            (1280,),
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
        view_3 = linear_6.view(1, -1, 16, 80)
        linear_6 = None
        key_layer_1 = view_3.transpose(1, 2)
        view_3 = None
        linear_7 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_4 = linear_7.view(1, -1, 16, 80)
        linear_7 = None
        value_layer_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_8 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_2 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_5 = linear_8.view(1, -1, 16, 80)
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
        context_layer_1 = attn_output_3.reshape((1, 257, 1280))
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
            (1280,),
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
            (1280,),
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
        view_6 = linear_12.view(1, -1, 16, 80)
        linear_12 = None
        key_layer_2 = view_6.transpose(1, 2)
        view_6 = None
        linear_13 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_7 = linear_13.view(1, -1, 16, 80)
        linear_13 = None
        value_layer_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_14 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_4 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_8 = linear_14.view(1, -1, 16, 80)
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
        context_layer_2 = attn_output_5.reshape((1, 257, 1280))
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
            (1280,),
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
            (1280,),
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
        view_9 = linear_18.view(1, -1, 16, 80)
        linear_18 = None
        key_layer_3 = view_9.transpose(1, 2)
        view_9 = None
        linear_19 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_10 = linear_19.view(1, -1, 16, 80)
        linear_19 = None
        value_layer_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_20 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_6 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_11 = linear_20.view(1, -1, 16, 80)
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
        context_layer_3 = attn_output_7.reshape((1, 257, 1280))
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
            (1280,),
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
            (1280,),
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
        view_12 = linear_24.view(1, -1, 16, 80)
        linear_24 = None
        key_layer_4 = view_12.transpose(1, 2)
        view_12 = None
        linear_25 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_13 = linear_25.view(1, -1, 16, 80)
        linear_25 = None
        value_layer_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_26 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_8 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_14 = linear_26.view(1, -1, 16, 80)
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
        context_layer_4 = attn_output_9.reshape((1, 257, 1280))
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
            (1280,),
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
            (1280,),
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
        view_15 = linear_30.view(1, -1, 16, 80)
        linear_30 = None
        key_layer_5 = view_15.transpose(1, 2)
        view_15 = None
        linear_31 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_16 = linear_31.view(1, -1, 16, 80)
        linear_31 = None
        value_layer_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_32 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_10 = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_17 = linear_32.view(1, -1, 16, 80)
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
        context_layer_5 = attn_output_11.reshape((1, 257, 1280))
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
            (1280,),
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
            (1280,),
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
        view_18 = linear_36.view(1, -1, 16, 80)
        linear_36 = None
        key_layer_6 = view_18.transpose(1, 2)
        view_18 = None
        linear_37 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_19 = linear_37.view(1, -1, 16, 80)
        linear_37 = None
        value_layer_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_38 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_12 = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_20 = linear_38.view(1, -1, 16, 80)
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
        context_layer_6 = attn_output_13.reshape((1, 257, 1280))
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
            (1280,),
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
            (1280,),
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
        view_21 = linear_42.view(1, -1, 16, 80)
        linear_42 = None
        key_layer_7 = view_21.transpose(1, 2)
        view_21 = None
        linear_43 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_22 = linear_43.view(1, -1, 16, 80)
        linear_43 = None
        value_layer_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_44 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_14 = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_23 = linear_44.view(1, -1, 16, 80)
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
        context_layer_7 = attn_output_15.reshape((1, 257, 1280))
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
            (1280,),
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
            (1280,),
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
        view_24 = linear_48.view(1, -1, 16, 80)
        linear_48 = None
        key_layer_8 = view_24.transpose(1, 2)
        view_24 = None
        linear_49 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_25 = linear_49.view(1, -1, 16, 80)
        linear_49 = None
        value_layer_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_50 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_26 = linear_50.view(1, -1, 16, 80)
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
        context_layer_8 = attn_output_17.reshape((1, 257, 1280))
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
            (1280,),
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
            (1280,),
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
        view_27 = linear_54.view(1, -1, 16, 80)
        linear_54 = None
        key_layer_9 = view_27.transpose(1, 2)
        view_27 = None
        linear_55 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_28 = linear_55.view(1, -1, 16, 80)
        linear_55 = None
        value_layer_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_56 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_18 = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_29 = linear_56.view(1, -1, 16, 80)
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
        context_layer_9 = attn_output_19.reshape((1, 257, 1280))
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
            (1280,),
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
            (1280,),
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
        view_30 = linear_60.view(1, -1, 16, 80)
        linear_60 = None
        key_layer_10 = view_30.transpose(1, 2)
        view_30 = None
        linear_61 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_31 = linear_61.view(1, -1, 16, 80)
        linear_61 = None
        value_layer_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_62 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_20 = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_32 = linear_62.view(1, -1, 16, 80)
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
        context_layer_10 = attn_output_21.reshape((1, 257, 1280))
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
            (1280,),
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
            (1280,),
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
        view_33 = linear_66.view(1, -1, 16, 80)
        linear_66 = None
        key_layer_11 = view_33.transpose(1, 2)
        view_33 = None
        linear_67 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_34 = linear_67.view(1, -1, 16, 80)
        linear_67 = None
        value_layer_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_68 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_22 = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_35 = linear_68.view(1, -1, 16, 80)
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
        context_layer_11 = attn_output_23.reshape((1, 257, 1280))
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
            (1280,),
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
        item_61 = (
            l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_eps = (
            None
        )
        layer_norm_24 = torch.nn.functional.layer_norm(
            hidden_states_95,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_bias_,
            item_61,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_bias_ = (item_61) = (
            None
        )
        linear_72 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_36 = linear_72.view(1, -1, 16, 80)
        linear_72 = None
        key_layer_12 = view_36.transpose(1, 2)
        view_36 = None
        linear_73 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_37 = linear_73.view(1, -1, 16, 80)
        linear_73 = None
        value_layer_12 = view_37.transpose(1, 2)
        view_37 = None
        linear_74 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_24 = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_38 = linear_74.view(1, -1, 16, 80)
        linear_74 = None
        query_layer_12 = view_38.transpose(1, 2)
        view_38 = None
        query_12 = query_layer_12.contiguous()
        query_layer_12 = None
        key_12 = key_layer_12.contiguous()
        key_layer_12 = None
        value_12 = value_layer_12.contiguous()
        value_layer_12 = None
        item_62 = (
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_12,
            key_12,
            value_12,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_62,
            is_causal=False,
        )
        query_12 = key_12 = value_12 = item_62 = None
        transpose_52 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_52.contiguous()
        transpose_52 = None
        context_layer_12 = attn_output_25.reshape((1, 257, 1280))
        attn_output_25 = None
        hidden_states_96 = torch._C._nn.linear(
            context_layer_12,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_12 = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_63 = (
            l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_97 = torch.nn.functional.dropout(
            hidden_states_96, item_63, False, False
        )
        hidden_states_96 = item_63 = None
        hidden_states_98 = hidden_states_97 + hidden_states_95
        hidden_states_97 = hidden_states_95 = None
        item_64 = (
            l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_eps = (
            None
        )
        layer_output_12 = torch.nn.functional.layer_norm(
            hidden_states_98,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_bias_,
            item_64,
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_bias_ = (item_64) = (
            None
        )
        hidden_states_99 = torch._C._nn.linear(
            layer_output_12,
            l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_12 = l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_100 = torch._C._nn.gelu(hidden_states_99)
        hidden_states_99 = None
        hidden_states_101 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_100 = l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_ = (None)
        item_65 = (
            l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_102 = torch.nn.functional.dropout(
            hidden_states_101, item_65, False, False
        )
        hidden_states_101 = item_65 = None
        hidden_states_103 = hidden_states_102 + hidden_states_98
        hidden_states_102 = hidden_states_98 = None
        item_66 = (
            l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_eps = (
            None
        )
        layer_norm_26 = torch.nn.functional.layer_norm(
            hidden_states_103,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_bias_,
            item_66,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_bias_ = (item_66) = (
            None
        )
        linear_78 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_39 = linear_78.view(1, -1, 16, 80)
        linear_78 = None
        key_layer_13 = view_39.transpose(1, 2)
        view_39 = None
        linear_79 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_40 = linear_79.view(1, -1, 16, 80)
        linear_79 = None
        value_layer_13 = view_40.transpose(1, 2)
        view_40 = None
        linear_80 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_26 = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_41 = linear_80.view(1, -1, 16, 80)
        linear_80 = None
        query_layer_13 = view_41.transpose(1, 2)
        view_41 = None
        query_13 = query_layer_13.contiguous()
        query_layer_13 = None
        key_13 = key_layer_13.contiguous()
        key_layer_13 = None
        value_13 = value_layer_13.contiguous()
        value_layer_13 = None
        item_67 = (
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_26 = torch._C._nn.scaled_dot_product_attention(
            query_13,
            key_13,
            value_13,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_67,
            is_causal=False,
        )
        query_13 = key_13 = value_13 = item_67 = None
        transpose_56 = attn_output_26.transpose(1, 2)
        attn_output_26 = None
        attn_output_27 = transpose_56.contiguous()
        transpose_56 = None
        context_layer_13 = attn_output_27.reshape((1, 257, 1280))
        attn_output_27 = None
        hidden_states_104 = torch._C._nn.linear(
            context_layer_13,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_13 = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_68 = (
            l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_105 = torch.nn.functional.dropout(
            hidden_states_104, item_68, False, False
        )
        hidden_states_104 = item_68 = None
        hidden_states_106 = hidden_states_105 + hidden_states_103
        hidden_states_105 = hidden_states_103 = None
        item_69 = (
            l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_eps = (
            None
        )
        layer_output_13 = torch.nn.functional.layer_norm(
            hidden_states_106,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_bias_,
            item_69,
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_bias_ = (item_69) = (
            None
        )
        hidden_states_107 = torch._C._nn.linear(
            layer_output_13,
            l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_13 = l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_108 = torch._C._nn.gelu(hidden_states_107)
        hidden_states_107 = None
        hidden_states_109 = torch._C._nn.linear(
            hidden_states_108,
            l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_108 = l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_ = (None)
        item_70 = (
            l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_110 = torch.nn.functional.dropout(
            hidden_states_109, item_70, False, False
        )
        hidden_states_109 = item_70 = None
        hidden_states_111 = hidden_states_110 + hidden_states_106
        hidden_states_110 = hidden_states_106 = None
        item_71 = (
            l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_eps = (
            None
        )
        layer_norm_28 = torch.nn.functional.layer_norm(
            hidden_states_111,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_bias_,
            item_71,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_bias_ = (item_71) = (
            None
        )
        linear_84 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_42 = linear_84.view(1, -1, 16, 80)
        linear_84 = None
        key_layer_14 = view_42.transpose(1, 2)
        view_42 = None
        linear_85 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_43 = linear_85.view(1, -1, 16, 80)
        linear_85 = None
        value_layer_14 = view_43.transpose(1, 2)
        view_43 = None
        linear_86 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_28 = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_44 = linear_86.view(1, -1, 16, 80)
        linear_86 = None
        query_layer_14 = view_44.transpose(1, 2)
        view_44 = None
        query_14 = query_layer_14.contiguous()
        query_layer_14 = None
        key_14 = key_layer_14.contiguous()
        key_layer_14 = None
        value_14 = value_layer_14.contiguous()
        value_layer_14 = None
        item_72 = (
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            query_14,
            key_14,
            value_14,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_72,
            is_causal=False,
        )
        query_14 = key_14 = value_14 = item_72 = None
        transpose_60 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_60.contiguous()
        transpose_60 = None
        context_layer_14 = attn_output_29.reshape((1, 257, 1280))
        attn_output_29 = None
        hidden_states_112 = torch._C._nn.linear(
            context_layer_14,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_14 = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_73 = (
            l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_113 = torch.nn.functional.dropout(
            hidden_states_112, item_73, False, False
        )
        hidden_states_112 = item_73 = None
        hidden_states_114 = hidden_states_113 + hidden_states_111
        hidden_states_113 = hidden_states_111 = None
        item_74 = (
            l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_eps = (
            None
        )
        layer_output_14 = torch.nn.functional.layer_norm(
            hidden_states_114,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_bias_,
            item_74,
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_bias_ = (item_74) = (
            None
        )
        hidden_states_115 = torch._C._nn.linear(
            layer_output_14,
            l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_14 = l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_116 = torch._C._nn.gelu(hidden_states_115)
        hidden_states_115 = None
        hidden_states_117 = torch._C._nn.linear(
            hidden_states_116,
            l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_116 = l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_ = (None)
        item_75 = (
            l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_118 = torch.nn.functional.dropout(
            hidden_states_117, item_75, False, False
        )
        hidden_states_117 = item_75 = None
        hidden_states_119 = hidden_states_118 + hidden_states_114
        hidden_states_118 = hidden_states_114 = None
        item_76 = (
            l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_eps = (
            None
        )
        layer_norm_30 = torch.nn.functional.layer_norm(
            hidden_states_119,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_bias_,
            item_76,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_bias_ = (item_76) = (
            None
        )
        linear_90 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_45 = linear_90.view(1, -1, 16, 80)
        linear_90 = None
        key_layer_15 = view_45.transpose(1, 2)
        view_45 = None
        linear_91 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_46 = linear_91.view(1, -1, 16, 80)
        linear_91 = None
        value_layer_15 = view_46.transpose(1, 2)
        view_46 = None
        linear_92 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_30 = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_47 = linear_92.view(1, -1, 16, 80)
        linear_92 = None
        query_layer_15 = view_47.transpose(1, 2)
        view_47 = None
        query_15 = query_layer_15.contiguous()
        query_layer_15 = None
        key_15 = key_layer_15.contiguous()
        key_layer_15 = None
        value_15 = value_layer_15.contiguous()
        value_layer_15 = None
        item_77 = (
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_30 = torch._C._nn.scaled_dot_product_attention(
            query_15,
            key_15,
            value_15,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_77,
            is_causal=False,
        )
        query_15 = key_15 = value_15 = item_77 = None
        transpose_64 = attn_output_30.transpose(1, 2)
        attn_output_30 = None
        attn_output_31 = transpose_64.contiguous()
        transpose_64 = None
        context_layer_15 = attn_output_31.reshape((1, 257, 1280))
        attn_output_31 = None
        hidden_states_120 = torch._C._nn.linear(
            context_layer_15,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_15 = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_78 = (
            l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_121 = torch.nn.functional.dropout(
            hidden_states_120, item_78, False, False
        )
        hidden_states_120 = item_78 = None
        hidden_states_122 = hidden_states_121 + hidden_states_119
        hidden_states_121 = hidden_states_119 = None
        item_79 = (
            l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_eps = (
            None
        )
        layer_output_15 = torch.nn.functional.layer_norm(
            hidden_states_122,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_bias_,
            item_79,
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_bias_ = (item_79) = (
            None
        )
        hidden_states_123 = torch._C._nn.linear(
            layer_output_15,
            l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_15 = l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_124 = torch._C._nn.gelu(hidden_states_123)
        hidden_states_123 = None
        hidden_states_125 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_124 = l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_ = (None)
        item_80 = (
            l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_126 = torch.nn.functional.dropout(
            hidden_states_125, item_80, False, False
        )
        hidden_states_125 = item_80 = None
        hidden_states_127 = hidden_states_126 + hidden_states_122
        hidden_states_126 = hidden_states_122 = None
        item_81 = (
            l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_eps = (
            None
        )
        layer_norm_32 = torch.nn.functional.layer_norm(
            hidden_states_127,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_bias_,
            item_81,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_bias_ = (item_81) = (
            None
        )
        linear_96 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_48 = linear_96.view(1, -1, 16, 80)
        linear_96 = None
        key_layer_16 = view_48.transpose(1, 2)
        view_48 = None
        linear_97 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_49 = linear_97.view(1, -1, 16, 80)
        linear_97 = None
        value_layer_16 = view_49.transpose(1, 2)
        view_49 = None
        linear_98 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_32 = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_50 = linear_98.view(1, -1, 16, 80)
        linear_98 = None
        query_layer_16 = view_50.transpose(1, 2)
        view_50 = None
        query_16 = query_layer_16.contiguous()
        query_layer_16 = None
        key_16 = key_layer_16.contiguous()
        key_layer_16 = None
        value_16 = value_layer_16.contiguous()
        value_layer_16 = None
        item_82 = (
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            query_16,
            key_16,
            value_16,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_82,
            is_causal=False,
        )
        query_16 = key_16 = value_16 = item_82 = None
        transpose_68 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_68.contiguous()
        transpose_68 = None
        context_layer_16 = attn_output_33.reshape((1, 257, 1280))
        attn_output_33 = None
        hidden_states_128 = torch._C._nn.linear(
            context_layer_16,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_16 = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_83 = (
            l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_129 = torch.nn.functional.dropout(
            hidden_states_128, item_83, False, False
        )
        hidden_states_128 = item_83 = None
        hidden_states_130 = hidden_states_129 + hidden_states_127
        hidden_states_129 = hidden_states_127 = None
        item_84 = (
            l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_eps = (
            None
        )
        layer_output_16 = torch.nn.functional.layer_norm(
            hidden_states_130,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_bias_,
            item_84,
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_bias_ = (item_84) = (
            None
        )
        hidden_states_131 = torch._C._nn.linear(
            layer_output_16,
            l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_16 = l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_132 = torch._C._nn.gelu(hidden_states_131)
        hidden_states_131 = None
        hidden_states_133 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_132 = l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_ = (None)
        item_85 = (
            l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_134 = torch.nn.functional.dropout(
            hidden_states_133, item_85, False, False
        )
        hidden_states_133 = item_85 = None
        hidden_states_135 = hidden_states_134 + hidden_states_130
        hidden_states_134 = hidden_states_130 = None
        item_86 = (
            l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_eps = (
            None
        )
        layer_norm_34 = torch.nn.functional.layer_norm(
            hidden_states_135,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_bias_,
            item_86,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_bias_ = (item_86) = (
            None
        )
        linear_102 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_51 = linear_102.view(1, -1, 16, 80)
        linear_102 = None
        key_layer_17 = view_51.transpose(1, 2)
        view_51 = None
        linear_103 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_52 = linear_103.view(1, -1, 16, 80)
        linear_103 = None
        value_layer_17 = view_52.transpose(1, 2)
        view_52 = None
        linear_104 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_34 = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_53 = linear_104.view(1, -1, 16, 80)
        linear_104 = None
        query_layer_17 = view_53.transpose(1, 2)
        view_53 = None
        query_17 = query_layer_17.contiguous()
        query_layer_17 = None
        key_17 = key_layer_17.contiguous()
        key_layer_17 = None
        value_17 = value_layer_17.contiguous()
        value_layer_17 = None
        item_87 = (
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_34 = torch._C._nn.scaled_dot_product_attention(
            query_17,
            key_17,
            value_17,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_87,
            is_causal=False,
        )
        query_17 = key_17 = value_17 = item_87 = None
        transpose_72 = attn_output_34.transpose(1, 2)
        attn_output_34 = None
        attn_output_35 = transpose_72.contiguous()
        transpose_72 = None
        context_layer_17 = attn_output_35.reshape((1, 257, 1280))
        attn_output_35 = None
        hidden_states_136 = torch._C._nn.linear(
            context_layer_17,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_17 = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_88 = (
            l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_137 = torch.nn.functional.dropout(
            hidden_states_136, item_88, False, False
        )
        hidden_states_136 = item_88 = None
        hidden_states_138 = hidden_states_137 + hidden_states_135
        hidden_states_137 = hidden_states_135 = None
        item_89 = (
            l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_eps = (
            None
        )
        layer_output_17 = torch.nn.functional.layer_norm(
            hidden_states_138,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_bias_,
            item_89,
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_bias_ = (item_89) = (
            None
        )
        hidden_states_139 = torch._C._nn.linear(
            layer_output_17,
            l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_17 = l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_140 = torch._C._nn.gelu(hidden_states_139)
        hidden_states_139 = None
        hidden_states_141 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_140 = l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_ = (None)
        item_90 = (
            l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_142 = torch.nn.functional.dropout(
            hidden_states_141, item_90, False, False
        )
        hidden_states_141 = item_90 = None
        hidden_states_143 = hidden_states_142 + hidden_states_138
        hidden_states_142 = hidden_states_138 = None
        item_91 = (
            l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_eps = (
            None
        )
        layer_norm_36 = torch.nn.functional.layer_norm(
            hidden_states_143,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_bias_,
            item_91,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_bias_ = (item_91) = (
            None
        )
        linear_108 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_54 = linear_108.view(1, -1, 16, 80)
        linear_108 = None
        key_layer_18 = view_54.transpose(1, 2)
        view_54 = None
        linear_109 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_55 = linear_109.view(1, -1, 16, 80)
        linear_109 = None
        value_layer_18 = view_55.transpose(1, 2)
        view_55 = None
        linear_110 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_36 = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_56 = linear_110.view(1, -1, 16, 80)
        linear_110 = None
        query_layer_18 = view_56.transpose(1, 2)
        view_56 = None
        query_18 = query_layer_18.contiguous()
        query_layer_18 = None
        key_18 = key_layer_18.contiguous()
        key_layer_18 = None
        value_18 = value_layer_18.contiguous()
        value_layer_18 = None
        item_92 = (
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_18,
            key_18,
            value_18,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_92,
            is_causal=False,
        )
        query_18 = key_18 = value_18 = item_92 = None
        transpose_76 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_76.contiguous()
        transpose_76 = None
        context_layer_18 = attn_output_37.reshape((1, 257, 1280))
        attn_output_37 = None
        hidden_states_144 = torch._C._nn.linear(
            context_layer_18,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_18 = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_93 = (
            l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_145 = torch.nn.functional.dropout(
            hidden_states_144, item_93, False, False
        )
        hidden_states_144 = item_93 = None
        hidden_states_146 = hidden_states_145 + hidden_states_143
        hidden_states_145 = hidden_states_143 = None
        item_94 = (
            l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_eps = (
            None
        )
        layer_output_18 = torch.nn.functional.layer_norm(
            hidden_states_146,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_bias_,
            item_94,
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_bias_ = (item_94) = (
            None
        )
        hidden_states_147 = torch._C._nn.linear(
            layer_output_18,
            l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_18 = l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_148 = torch._C._nn.gelu(hidden_states_147)
        hidden_states_147 = None
        hidden_states_149 = torch._C._nn.linear(
            hidden_states_148,
            l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_148 = l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_ = (None)
        item_95 = (
            l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_150 = torch.nn.functional.dropout(
            hidden_states_149, item_95, False, False
        )
        hidden_states_149 = item_95 = None
        hidden_states_151 = hidden_states_150 + hidden_states_146
        hidden_states_150 = hidden_states_146 = None
        item_96 = (
            l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_eps = (
            None
        )
        layer_norm_38 = torch.nn.functional.layer_norm(
            hidden_states_151,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_bias_,
            item_96,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_bias_ = (item_96) = (
            None
        )
        linear_114 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_57 = linear_114.view(1, -1, 16, 80)
        linear_114 = None
        key_layer_19 = view_57.transpose(1, 2)
        view_57 = None
        linear_115 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_58 = linear_115.view(1, -1, 16, 80)
        linear_115 = None
        value_layer_19 = view_58.transpose(1, 2)
        view_58 = None
        linear_116 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_38 = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_59 = linear_116.view(1, -1, 16, 80)
        linear_116 = None
        query_layer_19 = view_59.transpose(1, 2)
        view_59 = None
        query_19 = query_layer_19.contiguous()
        query_layer_19 = None
        key_19 = key_layer_19.contiguous()
        key_layer_19 = None
        value_19 = value_layer_19.contiguous()
        value_layer_19 = None
        item_97 = (
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_38 = torch._C._nn.scaled_dot_product_attention(
            query_19,
            key_19,
            value_19,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_97,
            is_causal=False,
        )
        query_19 = key_19 = value_19 = item_97 = None
        transpose_80 = attn_output_38.transpose(1, 2)
        attn_output_38 = None
        attn_output_39 = transpose_80.contiguous()
        transpose_80 = None
        context_layer_19 = attn_output_39.reshape((1, 257, 1280))
        attn_output_39 = None
        hidden_states_152 = torch._C._nn.linear(
            context_layer_19,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_19 = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_98 = (
            l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_153 = torch.nn.functional.dropout(
            hidden_states_152, item_98, False, False
        )
        hidden_states_152 = item_98 = None
        hidden_states_154 = hidden_states_153 + hidden_states_151
        hidden_states_153 = hidden_states_151 = None
        item_99 = (
            l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_eps = (
            None
        )
        layer_output_19 = torch.nn.functional.layer_norm(
            hidden_states_154,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_bias_,
            item_99,
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_bias_ = (item_99) = (
            None
        )
        hidden_states_155 = torch._C._nn.linear(
            layer_output_19,
            l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_19 = l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_156 = torch._C._nn.gelu(hidden_states_155)
        hidden_states_155 = None
        hidden_states_157 = torch._C._nn.linear(
            hidden_states_156,
            l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_156 = l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_ = (None)
        item_100 = (
            l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_158 = torch.nn.functional.dropout(
            hidden_states_157, item_100, False, False
        )
        hidden_states_157 = item_100 = None
        hidden_states_159 = hidden_states_158 + hidden_states_154
        hidden_states_158 = hidden_states_154 = None
        item_101 = (
            l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_eps = (
            None
        )
        layer_norm_40 = torch.nn.functional.layer_norm(
            hidden_states_159,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_bias_,
            item_101,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_bias_ = (item_101) = (
            None
        )
        linear_120 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_60 = linear_120.view(1, -1, 16, 80)
        linear_120 = None
        key_layer_20 = view_60.transpose(1, 2)
        view_60 = None
        linear_121 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_61 = linear_121.view(1, -1, 16, 80)
        linear_121 = None
        value_layer_20 = view_61.transpose(1, 2)
        view_61 = None
        linear_122 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_40 = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_62 = linear_122.view(1, -1, 16, 80)
        linear_122 = None
        query_layer_20 = view_62.transpose(1, 2)
        view_62 = None
        query_20 = query_layer_20.contiguous()
        query_layer_20 = None
        key_20 = key_layer_20.contiguous()
        key_layer_20 = None
        value_20 = value_layer_20.contiguous()
        value_layer_20 = None
        item_102 = (
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_20,
            key_20,
            value_20,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_102,
            is_causal=False,
        )
        query_20 = key_20 = value_20 = item_102 = None
        transpose_84 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_84.contiguous()
        transpose_84 = None
        context_layer_20 = attn_output_41.reshape((1, 257, 1280))
        attn_output_41 = None
        hidden_states_160 = torch._C._nn.linear(
            context_layer_20,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_20 = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_103 = (
            l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_161 = torch.nn.functional.dropout(
            hidden_states_160, item_103, False, False
        )
        hidden_states_160 = item_103 = None
        hidden_states_162 = hidden_states_161 + hidden_states_159
        hidden_states_161 = hidden_states_159 = None
        item_104 = (
            l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_eps = (
            None
        )
        layer_output_20 = torch.nn.functional.layer_norm(
            hidden_states_162,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_bias_,
            item_104,
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_bias_ = (item_104) = (
            None
        )
        hidden_states_163 = torch._C._nn.linear(
            layer_output_20,
            l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_20 = l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_164 = torch._C._nn.gelu(hidden_states_163)
        hidden_states_163 = None
        hidden_states_165 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_164 = l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_ = (None)
        item_105 = (
            l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_166 = torch.nn.functional.dropout(
            hidden_states_165, item_105, False, False
        )
        hidden_states_165 = item_105 = None
        hidden_states_167 = hidden_states_166 + hidden_states_162
        hidden_states_166 = hidden_states_162 = None
        item_106 = (
            l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_eps = (
            None
        )
        layer_norm_42 = torch.nn.functional.layer_norm(
            hidden_states_167,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_bias_,
            item_106,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_bias_ = (item_106) = (
            None
        )
        linear_126 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_63 = linear_126.view(1, -1, 16, 80)
        linear_126 = None
        key_layer_21 = view_63.transpose(1, 2)
        view_63 = None
        linear_127 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_64 = linear_127.view(1, -1, 16, 80)
        linear_127 = None
        value_layer_21 = view_64.transpose(1, 2)
        view_64 = None
        linear_128 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_42 = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_65 = linear_128.view(1, -1, 16, 80)
        linear_128 = None
        query_layer_21 = view_65.transpose(1, 2)
        view_65 = None
        query_21 = query_layer_21.contiguous()
        query_layer_21 = None
        key_21 = key_layer_21.contiguous()
        key_layer_21 = None
        value_21 = value_layer_21.contiguous()
        value_layer_21 = None
        item_107 = (
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_42 = torch._C._nn.scaled_dot_product_attention(
            query_21,
            key_21,
            value_21,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_107,
            is_causal=False,
        )
        query_21 = key_21 = value_21 = item_107 = None
        transpose_88 = attn_output_42.transpose(1, 2)
        attn_output_42 = None
        attn_output_43 = transpose_88.contiguous()
        transpose_88 = None
        context_layer_21 = attn_output_43.reshape((1, 257, 1280))
        attn_output_43 = None
        hidden_states_168 = torch._C._nn.linear(
            context_layer_21,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_21 = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_108 = (
            l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_169 = torch.nn.functional.dropout(
            hidden_states_168, item_108, False, False
        )
        hidden_states_168 = item_108 = None
        hidden_states_170 = hidden_states_169 + hidden_states_167
        hidden_states_169 = hidden_states_167 = None
        item_109 = (
            l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_eps = (
            None
        )
        layer_output_21 = torch.nn.functional.layer_norm(
            hidden_states_170,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_bias_,
            item_109,
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_bias_ = (item_109) = (
            None
        )
        hidden_states_171 = torch._C._nn.linear(
            layer_output_21,
            l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_21 = l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_172 = torch._C._nn.gelu(hidden_states_171)
        hidden_states_171 = None
        hidden_states_173 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_172 = l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_ = (None)
        item_110 = (
            l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_174 = torch.nn.functional.dropout(
            hidden_states_173, item_110, False, False
        )
        hidden_states_173 = item_110 = None
        hidden_states_175 = hidden_states_174 + hidden_states_170
        hidden_states_174 = hidden_states_170 = None
        item_111 = (
            l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_eps = (
            None
        )
        layer_norm_44 = torch.nn.functional.layer_norm(
            hidden_states_175,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_bias_,
            item_111,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_bias_ = (item_111) = (
            None
        )
        linear_132 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_66 = linear_132.view(1, -1, 16, 80)
        linear_132 = None
        key_layer_22 = view_66.transpose(1, 2)
        view_66 = None
        linear_133 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_67 = linear_133.view(1, -1, 16, 80)
        linear_133 = None
        value_layer_22 = view_67.transpose(1, 2)
        view_67 = None
        linear_134 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_44 = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_68 = linear_134.view(1, -1, 16, 80)
        linear_134 = None
        query_layer_22 = view_68.transpose(1, 2)
        view_68 = None
        query_22 = query_layer_22.contiguous()
        query_layer_22 = None
        key_22 = key_layer_22.contiguous()
        key_layer_22 = None
        value_22 = value_layer_22.contiguous()
        value_layer_22 = None
        item_112 = (
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            query_22,
            key_22,
            value_22,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_112,
            is_causal=False,
        )
        query_22 = key_22 = value_22 = item_112 = None
        transpose_92 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_92.contiguous()
        transpose_92 = None
        context_layer_22 = attn_output_45.reshape((1, 257, 1280))
        attn_output_45 = None
        hidden_states_176 = torch._C._nn.linear(
            context_layer_22,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_22 = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_113 = (
            l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_177 = torch.nn.functional.dropout(
            hidden_states_176, item_113, False, False
        )
        hidden_states_176 = item_113 = None
        hidden_states_178 = hidden_states_177 + hidden_states_175
        hidden_states_177 = hidden_states_175 = None
        item_114 = (
            l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_eps = (
            None
        )
        layer_output_22 = torch.nn.functional.layer_norm(
            hidden_states_178,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_bias_,
            item_114,
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_bias_ = (item_114) = (
            None
        )
        hidden_states_179 = torch._C._nn.linear(
            layer_output_22,
            l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_22 = l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_180 = torch._C._nn.gelu(hidden_states_179)
        hidden_states_179 = None
        hidden_states_181 = torch._C._nn.linear(
            hidden_states_180,
            l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_180 = l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_ = (None)
        item_115 = (
            l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_182 = torch.nn.functional.dropout(
            hidden_states_181, item_115, False, False
        )
        hidden_states_181 = item_115 = None
        hidden_states_183 = hidden_states_182 + hidden_states_178
        hidden_states_182 = hidden_states_178 = None
        item_116 = (
            l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_eps = (
            None
        )
        layer_norm_46 = torch.nn.functional.layer_norm(
            hidden_states_183,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_bias_,
            item_116,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_bias_ = (item_116) = (
            None
        )
        linear_138 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_69 = linear_138.view(1, -1, 16, 80)
        linear_138 = None
        key_layer_23 = view_69.transpose(1, 2)
        view_69 = None
        linear_139 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_70 = linear_139.view(1, -1, 16, 80)
        linear_139 = None
        value_layer_23 = view_70.transpose(1, 2)
        view_70 = None
        linear_140 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_46 = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_71 = linear_140.view(1, -1, 16, 80)
        linear_140 = None
        query_layer_23 = view_71.transpose(1, 2)
        view_71 = None
        query_23 = query_layer_23.contiguous()
        query_layer_23 = None
        key_23 = key_layer_23.contiguous()
        key_layer_23 = None
        value_23 = value_layer_23.contiguous()
        value_layer_23 = None
        item_117 = (
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_46 = torch._C._nn.scaled_dot_product_attention(
            query_23,
            key_23,
            value_23,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_117,
            is_causal=False,
        )
        query_23 = key_23 = value_23 = item_117 = None
        transpose_96 = attn_output_46.transpose(1, 2)
        attn_output_46 = None
        attn_output_47 = transpose_96.contiguous()
        transpose_96 = None
        context_layer_23 = attn_output_47.reshape((1, 257, 1280))
        attn_output_47 = None
        hidden_states_184 = torch._C._nn.linear(
            context_layer_23,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_23 = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_118 = (
            l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_185 = torch.nn.functional.dropout(
            hidden_states_184, item_118, False, False
        )
        hidden_states_184 = item_118 = None
        hidden_states_186 = hidden_states_185 + hidden_states_183
        hidden_states_185 = hidden_states_183 = None
        item_119 = (
            l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_eps = (
            None
        )
        layer_output_23 = torch.nn.functional.layer_norm(
            hidden_states_186,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_bias_,
            item_119,
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_bias_ = (item_119) = (
            None
        )
        hidden_states_187 = torch._C._nn.linear(
            layer_output_23,
            l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_23 = l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_188 = torch._C._nn.gelu(hidden_states_187)
        hidden_states_187 = None
        hidden_states_189 = torch._C._nn.linear(
            hidden_states_188,
            l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_188 = l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_ = (None)
        item_120 = (
            l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_190 = torch.nn.functional.dropout(
            hidden_states_189, item_120, False, False
        )
        hidden_states_189 = item_120 = None
        hidden_states_191 = hidden_states_190 + hidden_states_186
        hidden_states_190 = hidden_states_186 = None
        item_121 = (
            l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_eps = (
            None
        )
        layer_norm_48 = torch.nn.functional.layer_norm(
            hidden_states_191,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_parameters_bias_,
            item_121,
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_before_parameters_bias_ = (item_121) = (
            None
        )
        linear_144 = torch._C._nn.linear(
            layer_norm_48,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_72 = linear_144.view(1, -1, 16, 80)
        linear_144 = None
        key_layer_24 = view_72.transpose(1, 2)
        view_72 = None
        linear_145 = torch._C._nn.linear(
            layer_norm_48,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_73 = linear_145.view(1, -1, 16, 80)
        linear_145 = None
        value_layer_24 = view_73.transpose(1, 2)
        view_73 = None
        linear_146 = torch._C._nn.linear(
            layer_norm_48,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_48 = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_74 = linear_146.view(1, -1, 16, 80)
        linear_146 = None
        query_layer_24 = view_74.transpose(1, 2)
        view_74 = None
        query_24 = query_layer_24.contiguous()
        query_layer_24 = None
        key_24 = key_layer_24.contiguous()
        key_layer_24 = None
        value_24 = value_layer_24.contiguous()
        value_layer_24 = None
        item_122 = (
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            query_24,
            key_24,
            value_24,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_122,
            is_causal=False,
        )
        query_24 = key_24 = value_24 = item_122 = None
        transpose_100 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_100.contiguous()
        transpose_100 = None
        context_layer_24 = attn_output_49.reshape((1, 257, 1280))
        attn_output_49 = None
        hidden_states_192 = torch._C._nn.linear(
            context_layer_24,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_24 = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_123 = (
            l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_193 = torch.nn.functional.dropout(
            hidden_states_192, item_123, False, False
        )
        hidden_states_192 = item_123 = None
        hidden_states_194 = hidden_states_193 + hidden_states_191
        hidden_states_193 = hidden_states_191 = None
        item_124 = (
            l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_eps = (
            None
        )
        layer_output_24 = torch.nn.functional.layer_norm(
            hidden_states_194,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_parameters_bias_,
            item_124,
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_layernorm_after_parameters_bias_ = (item_124) = (
            None
        )
        hidden_states_195 = torch._C._nn.linear(
            layer_output_24,
            l_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_24 = l_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_196 = torch._C._nn.gelu(hidden_states_195)
        hidden_states_195 = None
        hidden_states_197 = torch._C._nn.linear(
            hidden_states_196,
            l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_196 = l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_bias_ = (None)
        item_125 = (
            l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_198 = torch.nn.functional.dropout(
            hidden_states_197, item_125, False, False
        )
        hidden_states_197 = item_125 = None
        hidden_states_199 = hidden_states_198 + hidden_states_194
        hidden_states_198 = hidden_states_194 = None
        item_126 = (
            l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_eps = (
            None
        )
        layer_norm_50 = torch.nn.functional.layer_norm(
            hidden_states_199,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_parameters_bias_,
            item_126,
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_before_parameters_bias_ = (item_126) = (
            None
        )
        linear_150 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_75 = linear_150.view(1, -1, 16, 80)
        linear_150 = None
        key_layer_25 = view_75.transpose(1, 2)
        view_75 = None
        linear_151 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_76 = linear_151.view(1, -1, 16, 80)
        linear_151 = None
        value_layer_25 = view_76.transpose(1, 2)
        view_76 = None
        linear_152 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_50 = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_77 = linear_152.view(1, -1, 16, 80)
        linear_152 = None
        query_layer_25 = view_77.transpose(1, 2)
        view_77 = None
        query_25 = query_layer_25.contiguous()
        query_layer_25 = None
        key_25 = key_layer_25.contiguous()
        key_layer_25 = None
        value_25 = value_layer_25.contiguous()
        value_layer_25 = None
        item_127 = (
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_50 = torch._C._nn.scaled_dot_product_attention(
            query_25,
            key_25,
            value_25,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_127,
            is_causal=False,
        )
        query_25 = key_25 = value_25 = item_127 = None
        transpose_104 = attn_output_50.transpose(1, 2)
        attn_output_50 = None
        attn_output_51 = transpose_104.contiguous()
        transpose_104 = None
        context_layer_25 = attn_output_51.reshape((1, 257, 1280))
        attn_output_51 = None
        hidden_states_200 = torch._C._nn.linear(
            context_layer_25,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_25 = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_128 = (
            l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_201 = torch.nn.functional.dropout(
            hidden_states_200, item_128, False, False
        )
        hidden_states_200 = item_128 = None
        hidden_states_202 = hidden_states_201 + hidden_states_199
        hidden_states_201 = hidden_states_199 = None
        item_129 = (
            l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_eps = (
            None
        )
        layer_output_25 = torch.nn.functional.layer_norm(
            hidden_states_202,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_parameters_bias_,
            item_129,
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_layernorm_after_parameters_bias_ = (item_129) = (
            None
        )
        hidden_states_203 = torch._C._nn.linear(
            layer_output_25,
            l_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_25 = l_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_204 = torch._C._nn.gelu(hidden_states_203)
        hidden_states_203 = None
        hidden_states_205 = torch._C._nn.linear(
            hidden_states_204,
            l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_204 = l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_bias_ = (None)
        item_130 = (
            l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_206 = torch.nn.functional.dropout(
            hidden_states_205, item_130, False, False
        )
        hidden_states_205 = item_130 = None
        hidden_states_207 = hidden_states_206 + hidden_states_202
        hidden_states_206 = hidden_states_202 = None
        item_131 = (
            l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_eps = (
            None
        )
        layer_norm_52 = torch.nn.functional.layer_norm(
            hidden_states_207,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_parameters_bias_,
            item_131,
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_before_parameters_bias_ = (item_131) = (
            None
        )
        linear_156 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_78 = linear_156.view(1, -1, 16, 80)
        linear_156 = None
        key_layer_26 = view_78.transpose(1, 2)
        view_78 = None
        linear_157 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_79 = linear_157.view(1, -1, 16, 80)
        linear_157 = None
        value_layer_26 = view_79.transpose(1, 2)
        view_79 = None
        linear_158 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_52 = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_80 = linear_158.view(1, -1, 16, 80)
        linear_158 = None
        query_layer_26 = view_80.transpose(1, 2)
        view_80 = None
        query_26 = query_layer_26.contiguous()
        query_layer_26 = None
        key_26 = key_layer_26.contiguous()
        key_layer_26 = None
        value_26 = value_layer_26.contiguous()
        value_layer_26 = None
        item_132 = (
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_52 = torch._C._nn.scaled_dot_product_attention(
            query_26,
            key_26,
            value_26,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_132,
            is_causal=False,
        )
        query_26 = key_26 = value_26 = item_132 = None
        transpose_108 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_108.contiguous()
        transpose_108 = None
        context_layer_26 = attn_output_53.reshape((1, 257, 1280))
        attn_output_53 = None
        hidden_states_208 = torch._C._nn.linear(
            context_layer_26,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_26 = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_133 = (
            l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_209 = torch.nn.functional.dropout(
            hidden_states_208, item_133, False, False
        )
        hidden_states_208 = item_133 = None
        hidden_states_210 = hidden_states_209 + hidden_states_207
        hidden_states_209 = hidden_states_207 = None
        item_134 = (
            l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_eps = (
            None
        )
        layer_output_26 = torch.nn.functional.layer_norm(
            hidden_states_210,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_parameters_bias_,
            item_134,
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_layernorm_after_parameters_bias_ = (item_134) = (
            None
        )
        hidden_states_211 = torch._C._nn.linear(
            layer_output_26,
            l_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_26 = l_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_212 = torch._C._nn.gelu(hidden_states_211)
        hidden_states_211 = None
        hidden_states_213 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_212 = l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_bias_ = (None)
        item_135 = (
            l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_214 = torch.nn.functional.dropout(
            hidden_states_213, item_135, False, False
        )
        hidden_states_213 = item_135 = None
        hidden_states_215 = hidden_states_214 + hidden_states_210
        hidden_states_214 = hidden_states_210 = None
        item_136 = (
            l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_eps = (
            None
        )
        layer_norm_54 = torch.nn.functional.layer_norm(
            hidden_states_215,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_parameters_bias_,
            item_136,
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_before_parameters_bias_ = (item_136) = (
            None
        )
        linear_162 = torch._C._nn.linear(
            layer_norm_54,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_81 = linear_162.view(1, -1, 16, 80)
        linear_162 = None
        key_layer_27 = view_81.transpose(1, 2)
        view_81 = None
        linear_163 = torch._C._nn.linear(
            layer_norm_54,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_82 = linear_163.view(1, -1, 16, 80)
        linear_163 = None
        value_layer_27 = view_82.transpose(1, 2)
        view_82 = None
        linear_164 = torch._C._nn.linear(
            layer_norm_54,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_54 = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_83 = linear_164.view(1, -1, 16, 80)
        linear_164 = None
        query_layer_27 = view_83.transpose(1, 2)
        view_83 = None
        query_27 = query_layer_27.contiguous()
        query_layer_27 = None
        key_27 = key_layer_27.contiguous()
        key_layer_27 = None
        value_27 = value_layer_27.contiguous()
        value_layer_27 = None
        item_137 = (
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_54 = torch._C._nn.scaled_dot_product_attention(
            query_27,
            key_27,
            value_27,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_137,
            is_causal=False,
        )
        query_27 = key_27 = value_27 = item_137 = None
        transpose_112 = attn_output_54.transpose(1, 2)
        attn_output_54 = None
        attn_output_55 = transpose_112.contiguous()
        transpose_112 = None
        context_layer_27 = attn_output_55.reshape((1, 257, 1280))
        attn_output_55 = None
        hidden_states_216 = torch._C._nn.linear(
            context_layer_27,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_27 = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_138 = (
            l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_217 = torch.nn.functional.dropout(
            hidden_states_216, item_138, False, False
        )
        hidden_states_216 = item_138 = None
        hidden_states_218 = hidden_states_217 + hidden_states_215
        hidden_states_217 = hidden_states_215 = None
        item_139 = (
            l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_eps = (
            None
        )
        layer_output_27 = torch.nn.functional.layer_norm(
            hidden_states_218,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_parameters_bias_,
            item_139,
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_layernorm_after_parameters_bias_ = (item_139) = (
            None
        )
        hidden_states_219 = torch._C._nn.linear(
            layer_output_27,
            l_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_27 = l_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_220 = torch._C._nn.gelu(hidden_states_219)
        hidden_states_219 = None
        hidden_states_221 = torch._C._nn.linear(
            hidden_states_220,
            l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_220 = l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_bias_ = (None)
        item_140 = (
            l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_222 = torch.nn.functional.dropout(
            hidden_states_221, item_140, False, False
        )
        hidden_states_221 = item_140 = None
        hidden_states_223 = hidden_states_222 + hidden_states_218
        hidden_states_222 = hidden_states_218 = None
        item_141 = (
            l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_eps = (
            None
        )
        layer_norm_56 = torch.nn.functional.layer_norm(
            hidden_states_223,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_parameters_bias_,
            item_141,
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_before_parameters_bias_ = (item_141) = (
            None
        )
        linear_168 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_84 = linear_168.view(1, -1, 16, 80)
        linear_168 = None
        key_layer_28 = view_84.transpose(1, 2)
        view_84 = None
        linear_169 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_85 = linear_169.view(1, -1, 16, 80)
        linear_169 = None
        value_layer_28 = view_85.transpose(1, 2)
        view_85 = None
        linear_170 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_56 = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_86 = linear_170.view(1, -1, 16, 80)
        linear_170 = None
        query_layer_28 = view_86.transpose(1, 2)
        view_86 = None
        query_28 = query_layer_28.contiguous()
        query_layer_28 = None
        key_28 = key_layer_28.contiguous()
        key_layer_28 = None
        value_28 = value_layer_28.contiguous()
        value_layer_28 = None
        item_142 = (
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_56 = torch._C._nn.scaled_dot_product_attention(
            query_28,
            key_28,
            value_28,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_142,
            is_causal=False,
        )
        query_28 = key_28 = value_28 = item_142 = None
        transpose_116 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_116.contiguous()
        transpose_116 = None
        context_layer_28 = attn_output_57.reshape((1, 257, 1280))
        attn_output_57 = None
        hidden_states_224 = torch._C._nn.linear(
            context_layer_28,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_28 = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_143 = (
            l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_225 = torch.nn.functional.dropout(
            hidden_states_224, item_143, False, False
        )
        hidden_states_224 = item_143 = None
        hidden_states_226 = hidden_states_225 + hidden_states_223
        hidden_states_225 = hidden_states_223 = None
        item_144 = (
            l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_eps = (
            None
        )
        layer_output_28 = torch.nn.functional.layer_norm(
            hidden_states_226,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_parameters_bias_,
            item_144,
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_layernorm_after_parameters_bias_ = (item_144) = (
            None
        )
        hidden_states_227 = torch._C._nn.linear(
            layer_output_28,
            l_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_28 = l_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_228 = torch._C._nn.gelu(hidden_states_227)
        hidden_states_227 = None
        hidden_states_229 = torch._C._nn.linear(
            hidden_states_228,
            l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_228 = l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_bias_ = (None)
        item_145 = (
            l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_230 = torch.nn.functional.dropout(
            hidden_states_229, item_145, False, False
        )
        hidden_states_229 = item_145 = None
        hidden_states_231 = hidden_states_230 + hidden_states_226
        hidden_states_230 = hidden_states_226 = None
        item_146 = (
            l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_eps = (
            None
        )
        layer_norm_58 = torch.nn.functional.layer_norm(
            hidden_states_231,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_parameters_bias_,
            item_146,
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_before_parameters_bias_ = (item_146) = (
            None
        )
        linear_174 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_87 = linear_174.view(1, -1, 16, 80)
        linear_174 = None
        key_layer_29 = view_87.transpose(1, 2)
        view_87 = None
        linear_175 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_88 = linear_175.view(1, -1, 16, 80)
        linear_175 = None
        value_layer_29 = view_88.transpose(1, 2)
        view_88 = None
        linear_176 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_58 = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_89 = linear_176.view(1, -1, 16, 80)
        linear_176 = None
        query_layer_29 = view_89.transpose(1, 2)
        view_89 = None
        query_29 = query_layer_29.contiguous()
        query_layer_29 = None
        key_29 = key_layer_29.contiguous()
        key_layer_29 = None
        value_29 = value_layer_29.contiguous()
        value_layer_29 = None
        item_147 = (
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_58 = torch._C._nn.scaled_dot_product_attention(
            query_29,
            key_29,
            value_29,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_147,
            is_causal=False,
        )
        query_29 = key_29 = value_29 = item_147 = None
        transpose_120 = attn_output_58.transpose(1, 2)
        attn_output_58 = None
        attn_output_59 = transpose_120.contiguous()
        transpose_120 = None
        context_layer_29 = attn_output_59.reshape((1, 257, 1280))
        attn_output_59 = None
        hidden_states_232 = torch._C._nn.linear(
            context_layer_29,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_29 = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_148 = (
            l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_233 = torch.nn.functional.dropout(
            hidden_states_232, item_148, False, False
        )
        hidden_states_232 = item_148 = None
        hidden_states_234 = hidden_states_233 + hidden_states_231
        hidden_states_233 = hidden_states_231 = None
        item_149 = (
            l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_eps = (
            None
        )
        layer_output_29 = torch.nn.functional.layer_norm(
            hidden_states_234,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_parameters_bias_,
            item_149,
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_layernorm_after_parameters_bias_ = (item_149) = (
            None
        )
        hidden_states_235 = torch._C._nn.linear(
            layer_output_29,
            l_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_29 = l_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_236 = torch._C._nn.gelu(hidden_states_235)
        hidden_states_235 = None
        hidden_states_237 = torch._C._nn.linear(
            hidden_states_236,
            l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_236 = l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_bias_ = (None)
        item_150 = (
            l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_238 = torch.nn.functional.dropout(
            hidden_states_237, item_150, False, False
        )
        hidden_states_237 = item_150 = None
        hidden_states_239 = hidden_states_238 + hidden_states_234
        hidden_states_238 = hidden_states_234 = None
        item_151 = (
            l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_eps = (
            None
        )
        layer_norm_60 = torch.nn.functional.layer_norm(
            hidden_states_239,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_parameters_bias_,
            item_151,
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_before_parameters_bias_ = (item_151) = (
            None
        )
        linear_180 = torch._C._nn.linear(
            layer_norm_60,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_90 = linear_180.view(1, -1, 16, 80)
        linear_180 = None
        key_layer_30 = view_90.transpose(1, 2)
        view_90 = None
        linear_181 = torch._C._nn.linear(
            layer_norm_60,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_91 = linear_181.view(1, -1, 16, 80)
        linear_181 = None
        value_layer_30 = view_91.transpose(1, 2)
        view_91 = None
        linear_182 = torch._C._nn.linear(
            layer_norm_60,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_60 = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_92 = linear_182.view(1, -1, 16, 80)
        linear_182 = None
        query_layer_30 = view_92.transpose(1, 2)
        view_92 = None
        query_30 = query_layer_30.contiguous()
        query_layer_30 = None
        key_30 = key_layer_30.contiguous()
        key_layer_30 = None
        value_30 = value_layer_30.contiguous()
        value_layer_30 = None
        item_152 = (
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_30,
            key_30,
            value_30,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_152,
            is_causal=False,
        )
        query_30 = key_30 = value_30 = item_152 = None
        transpose_124 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_124.contiguous()
        transpose_124 = None
        context_layer_30 = attn_output_61.reshape((1, 257, 1280))
        attn_output_61 = None
        hidden_states_240 = torch._C._nn.linear(
            context_layer_30,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_30 = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_153 = (
            l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_241 = torch.nn.functional.dropout(
            hidden_states_240, item_153, False, False
        )
        hidden_states_240 = item_153 = None
        hidden_states_242 = hidden_states_241 + hidden_states_239
        hidden_states_241 = hidden_states_239 = None
        item_154 = (
            l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_eps = (
            None
        )
        layer_output_30 = torch.nn.functional.layer_norm(
            hidden_states_242,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_parameters_bias_,
            item_154,
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_layernorm_after_parameters_bias_ = (item_154) = (
            None
        )
        hidden_states_243 = torch._C._nn.linear(
            layer_output_30,
            l_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_30 = l_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_244 = torch._C._nn.gelu(hidden_states_243)
        hidden_states_243 = None
        hidden_states_245 = torch._C._nn.linear(
            hidden_states_244,
            l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_244 = l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_bias_ = (None)
        item_155 = (
            l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_246 = torch.nn.functional.dropout(
            hidden_states_245, item_155, False, False
        )
        hidden_states_245 = item_155 = None
        hidden_states_247 = hidden_states_246 + hidden_states_242
        hidden_states_246 = hidden_states_242 = None
        item_156 = (
            l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_eps = (
            None
        )
        layer_norm_62 = torch.nn.functional.layer_norm(
            hidden_states_247,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_parameters_bias_,
            item_156,
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_before_parameters_bias_ = (item_156) = (
            None
        )
        linear_186 = torch._C._nn.linear(
            layer_norm_62,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_93 = linear_186.view(1, -1, 16, 80)
        linear_186 = None
        key_layer_31 = view_93.transpose(1, 2)
        view_93 = None
        linear_187 = torch._C._nn.linear(
            layer_norm_62,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_94 = linear_187.view(1, -1, 16, 80)
        linear_187 = None
        value_layer_31 = view_94.transpose(1, 2)
        view_94 = None
        linear_188 = torch._C._nn.linear(
            layer_norm_62,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_62 = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_95 = linear_188.view(1, -1, 16, 80)
        linear_188 = None
        query_layer_31 = view_95.transpose(1, 2)
        view_95 = None
        query_31 = query_layer_31.contiguous()
        query_layer_31 = None
        key_31 = key_layer_31.contiguous()
        key_layer_31 = None
        value_31 = value_layer_31.contiguous()
        value_layer_31 = None
        item_157 = (
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_scaling = (
            None
        )
        attn_output_62 = torch._C._nn.scaled_dot_product_attention(
            query_31,
            key_31,
            value_31,
            attn_mask=None,
            dropout_p=0.0,
            scale=item_157,
            is_causal=False,
        )
        query_31 = key_31 = value_31 = item_157 = None
        transpose_128 = attn_output_62.transpose(1, 2)
        attn_output_62 = None
        attn_output_63 = transpose_128.contiguous()
        transpose_128 = None
        context_layer_31 = attn_output_63.reshape((1, 257, 1280))
        attn_output_63 = None
        hidden_states_248 = torch._C._nn.linear(
            context_layer_31,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_31 = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_158 = (
            l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_249 = torch.nn.functional.dropout(
            hidden_states_248, item_158, False, False
        )
        hidden_states_248 = item_158 = None
        hidden_states_250 = hidden_states_249 + hidden_states_247
        hidden_states_249 = hidden_states_247 = None
        item_159 = (
            l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_eps = (
            None
        )
        layer_output_31 = torch.nn.functional.layer_norm(
            hidden_states_250,
            (1280,),
            l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_parameters_bias_,
            item_159,
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_layernorm_after_parameters_bias_ = (item_159) = (
            None
        )
        hidden_states_251 = torch._C._nn.linear(
            layer_output_31,
            l_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_31 = l_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_252 = torch._C._nn.gelu(hidden_states_251)
        hidden_states_251 = None
        hidden_states_253 = torch._C._nn.linear(
            hidden_states_252,
            l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_252 = l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_bias_ = (None)
        item_160 = (
            l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_254 = torch.nn.functional.dropout(
            hidden_states_253, item_160, False, False
        )
        hidden_states_253 = item_160 = None
        hidden_states_255 = hidden_states_254 + hidden_states_250
        hidden_states_254 = hidden_states_250 = None
        item_161 = l_self_modules_layernorm_eps.item()
        l_self_modules_layernorm_eps = None
        sequence_output = torch.nn.functional.layer_norm(
            hidden_states_255,
            (1280,),
            l_self_modules_layernorm_parameters_weight_,
            l_self_modules_layernorm_parameters_bias_,
            item_161,
        )
        hidden_states_255 = (
            l_self_modules_layernorm_parameters_weight_
        ) = l_self_modules_layernorm_parameters_bias_ = item_161 = None
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
