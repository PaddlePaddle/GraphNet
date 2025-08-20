import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_pixel_values_: torch.Tensor,
        L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_parameters_detection_tokens_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_parameters_position_embeddings_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_parameters_mid_position_embeddings_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pooler_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_pixel_values_ = L_pixel_values_
        l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_ = L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_
        l_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_ = L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_
        l_self_modules_embeddings_parameters_cls_token_ = (
            L_self_modules_embeddings_parameters_cls_token_
        )
        l_self_modules_embeddings_parameters_detection_tokens_ = (
            L_self_modules_embeddings_parameters_detection_tokens_
        )
        l_self_modules_embeddings_parameters_position_embeddings_ = (
            L_self_modules_embeddings_parameters_position_embeddings_
        )
        l_self_modules_embeddings_modules_dropout_p = (
            L_self_modules_embeddings_modules_dropout_p
        )
        l_self_modules_encoder_parameters_mid_position_embeddings_ = (
            L_self_modules_encoder_parameters_mid_position_embeddings_
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_
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
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p = L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p
        l_self_modules_layernorm_parameters_weight_ = (
            L_self_modules_layernorm_parameters_weight_
        )
        l_self_modules_layernorm_parameters_bias_ = (
            L_self_modules_layernorm_parameters_bias_
        )
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
            (2, 2),
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
        detection_tokens = (
            l_self_modules_embeddings_parameters_detection_tokens_.expand(1, -1, -1)
        )
        l_self_modules_embeddings_parameters_detection_tokens_ = None
        embeddings_1 = torch.cat((cls_tokens, embeddings, detection_tokens), dim=1)
        cls_tokens = embeddings = detection_tokens = None
        cls_pos_embed = l_self_modules_embeddings_parameters_position_embeddings_[
            (slice(None, None, None), 0, slice(None, None, None))
        ]
        cls_pos_embed_1 = cls_pos_embed[(slice(None, None, None), None)]
        cls_pos_embed = None
        det_pos_embed = l_self_modules_embeddings_parameters_position_embeddings_[
            (slice(None, None, None), slice(-10, None, None), slice(None, None, None))
        ]
        patch_pos_embed = l_self_modules_embeddings_parameters_position_embeddings_[
            (slice(None, None, None), slice(1, -10, None), slice(None, None, None))
        ]
        l_self_modules_embeddings_parameters_position_embeddings_ = None
        patch_pos_embed_1 = patch_pos_embed.transpose(1, 2)
        patch_pos_embed = None
        patch_pos_embed_2 = patch_pos_embed_1.view(1, 32, 15, 15)
        patch_pos_embed_1 = None
        patch_pos_embed_3 = torch.nn.functional.interpolate(
            patch_pos_embed_2, size=(15, 15), mode="bicubic", align_corners=False
        )
        patch_pos_embed_2 = None
        flatten_1 = patch_pos_embed_3.flatten(2)
        patch_pos_embed_3 = None
        patch_pos_embed_4 = flatten_1.transpose(1, 2)
        flatten_1 = None
        scale_pos_embed = torch.cat(
            (cls_pos_embed_1, patch_pos_embed_4, det_pos_embed), dim=1
        )
        cls_pos_embed_1 = patch_pos_embed_4 = det_pos_embed = None
        embeddings_2 = embeddings_1 + scale_pos_embed
        embeddings_1 = scale_pos_embed = None
        item = l_self_modules_embeddings_modules_dropout_p.item()
        l_self_modules_embeddings_modules_dropout_p = None
        embeddings_3 = torch.nn.functional.dropout(embeddings_2, item, False, False)
        embeddings_2 = item = None
        cls_pos_embed_2 = l_self_modules_encoder_parameters_mid_position_embeddings_[
            (
                slice(None, None, None),
                slice(None, None, None),
                0,
                slice(None, None, None),
            )
        ]
        cls_pos_embed_3 = cls_pos_embed_2[(slice(None, None, None), None)]
        cls_pos_embed_2 = None
        det_pos_embed_1 = l_self_modules_encoder_parameters_mid_position_embeddings_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(-10, None, None),
                slice(None, None, None),
            )
        ]
        patch_pos_embed_5 = l_self_modules_encoder_parameters_mid_position_embeddings_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, -10, None),
                slice(None, None, None),
            )
        ]
        l_self_modules_encoder_parameters_mid_position_embeddings_ = None
        patch_pos_embed_6 = patch_pos_embed_5.transpose(2, 3)
        patch_pos_embed_5 = None
        patch_pos_embed_7 = patch_pos_embed_6.view(4, 32, 15, 15)
        patch_pos_embed_6 = None
        patch_pos_embed_8 = torch.nn.functional.interpolate(
            patch_pos_embed_7, size=(15, 15), mode="bicubic", align_corners=False
        )
        patch_pos_embed_7 = None
        flatten_2 = patch_pos_embed_8.flatten(2)
        patch_pos_embed_8 = None
        transpose_4 = flatten_2.transpose(1, 2)
        flatten_2 = None
        contiguous = transpose_4.contiguous()
        transpose_4 = None
        patch_pos_embed_9 = contiguous.view(4, 1, 225, 32)
        contiguous = None
        scale_pos_embed_1 = torch.cat(
            (cls_pos_embed_3, patch_pos_embed_9, det_pos_embed_1), dim=2
        )
        cls_pos_embed_3 = patch_pos_embed_9 = det_pos_embed_1 = None
        layer_norm = torch.nn.functional.layer_norm(
            embeddings_3,
            (32,),
            l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_ = (None)
        linear = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_3 = linear.view(1, -1, 4, 8)
        linear = None
        key_layer = view_3.transpose(1, 2)
        view_3 = None
        linear_1 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_4 = linear_1.view(1, -1, 4, 8)
        linear_1 = None
        value_layer = view_4.transpose(1, 2)
        view_4 = None
        linear_2 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_5 = linear_2.view(1, -1, 4, 8)
        linear_2 = None
        query_layer = view_5.transpose(1, 2)
        view_5 = None
        query = query_layer.contiguous()
        query_layer = None
        key = key_layer.contiguous()
        key_layer = None
        value = value_layer.contiguous()
        value_layer = None
        item_1 = (
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
            scale=item_1,
            is_causal=False,
        )
        query = key = value = item_1 = None
        transpose_8 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_8.contiguous()
        transpose_8 = None
        context_layer = attn_output_1.reshape((1, 236, 32))
        attn_output_1 = None
        hidden_states = torch._C._nn.linear(
            context_layer,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_2 = (
            l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_1 = torch.nn.functional.dropout(
            hidden_states, item_2, False, False
        )
        hidden_states = item_2 = None
        hidden_states_2 = hidden_states_1 + embeddings_3
        hidden_states_1 = embeddings_3 = None
        layer_output = torch.nn.functional.layer_norm(
            hidden_states_2,
            (32,),
            l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_ = (None)
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
        item_3 = (
            l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, item_3, False, False
        )
        hidden_states_5 = item_3 = None
        hidden_states_7 = hidden_states_6 + hidden_states_2
        hidden_states_6 = hidden_states_2 = None
        getitem_8 = scale_pos_embed_1[0]
        hidden_states_8 = hidden_states_7 + getitem_8
        hidden_states_7 = getitem_8 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            hidden_states_8,
            (32,),
            l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_6 = linear_6.view(1, -1, 4, 8)
        linear_6 = None
        key_layer_1 = view_6.transpose(1, 2)
        view_6 = None
        linear_7 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_7 = linear_7.view(1, -1, 4, 8)
        linear_7 = None
        value_layer_1 = view_7.transpose(1, 2)
        view_7 = None
        linear_8 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_2 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_8 = linear_8.view(1, -1, 4, 8)
        linear_8 = None
        query_layer_1 = view_8.transpose(1, 2)
        view_8 = None
        query_1 = query_layer_1.contiguous()
        query_layer_1 = None
        key_1 = key_layer_1.contiguous()
        key_layer_1 = None
        value_1 = value_layer_1.contiguous()
        value_layer_1 = None
        item_4 = (
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
            scale=item_4,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = item_4 = None
        transpose_12 = attn_output_2.transpose(1, 2)
        attn_output_2 = None
        attn_output_3 = transpose_12.contiguous()
        transpose_12 = None
        context_layer_1 = attn_output_3.reshape((1, 236, 32))
        attn_output_3 = None
        hidden_states_9 = torch._C._nn.linear(
            context_layer_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_1 = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_5 = (
            l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_10 = torch.nn.functional.dropout(
            hidden_states_9, item_5, False, False
        )
        hidden_states_9 = item_5 = None
        hidden_states_11 = hidden_states_10 + hidden_states_8
        hidden_states_10 = hidden_states_8 = None
        layer_output_1 = torch.nn.functional.layer_norm(
            hidden_states_11,
            (32,),
            l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_12 = torch._C._nn.linear(
            layer_output_1,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_1 = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_13 = torch._C._nn.gelu(hidden_states_12)
        hidden_states_12 = None
        hidden_states_14 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_13 = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        item_6 = (
            l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_15 = torch.nn.functional.dropout(
            hidden_states_14, item_6, False, False
        )
        hidden_states_14 = item_6 = None
        hidden_states_16 = hidden_states_15 + hidden_states_11
        hidden_states_15 = hidden_states_11 = None
        getitem_9 = scale_pos_embed_1[1]
        hidden_states_17 = hidden_states_16 + getitem_9
        hidden_states_16 = getitem_9 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            hidden_states_17,
            (32,),
            l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_9 = linear_12.view(1, -1, 4, 8)
        linear_12 = None
        key_layer_2 = view_9.transpose(1, 2)
        view_9 = None
        linear_13 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_10 = linear_13.view(1, -1, 4, 8)
        linear_13 = None
        value_layer_2 = view_10.transpose(1, 2)
        view_10 = None
        linear_14 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_4 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_11 = linear_14.view(1, -1, 4, 8)
        linear_14 = None
        query_layer_2 = view_11.transpose(1, 2)
        view_11 = None
        query_2 = query_layer_2.contiguous()
        query_layer_2 = None
        key_2 = key_layer_2.contiguous()
        key_layer_2 = None
        value_2 = value_layer_2.contiguous()
        value_layer_2 = None
        item_7 = (
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
            scale=item_7,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = item_7 = None
        transpose_16 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_16.contiguous()
        transpose_16 = None
        context_layer_2 = attn_output_5.reshape((1, 236, 32))
        attn_output_5 = None
        hidden_states_18 = torch._C._nn.linear(
            context_layer_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_8 = (
            l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_19 = torch.nn.functional.dropout(
            hidden_states_18, item_8, False, False
        )
        hidden_states_18 = item_8 = None
        hidden_states_20 = hidden_states_19 + hidden_states_17
        hidden_states_19 = hidden_states_17 = None
        layer_output_2 = torch.nn.functional.layer_norm(
            hidden_states_20,
            (32,),
            l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_21 = torch._C._nn.linear(
            layer_output_2,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_2 = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_22 = torch._C._nn.gelu(hidden_states_21)
        hidden_states_21 = None
        hidden_states_23 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_22 = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        item_9 = (
            l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_24 = torch.nn.functional.dropout(
            hidden_states_23, item_9, False, False
        )
        hidden_states_23 = item_9 = None
        hidden_states_25 = hidden_states_24 + hidden_states_20
        hidden_states_24 = hidden_states_20 = None
        getitem_10 = scale_pos_embed_1[2]
        hidden_states_26 = hidden_states_25 + getitem_10
        hidden_states_25 = getitem_10 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            hidden_states_26,
            (32,),
            l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_12 = linear_18.view(1, -1, 4, 8)
        linear_18 = None
        key_layer_3 = view_12.transpose(1, 2)
        view_12 = None
        linear_19 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_13 = linear_19.view(1, -1, 4, 8)
        linear_19 = None
        value_layer_3 = view_13.transpose(1, 2)
        view_13 = None
        linear_20 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_6 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_14 = linear_20.view(1, -1, 4, 8)
        linear_20 = None
        query_layer_3 = view_14.transpose(1, 2)
        view_14 = None
        query_3 = query_layer_3.contiguous()
        query_layer_3 = None
        key_3 = key_layer_3.contiguous()
        key_layer_3 = None
        value_3 = value_layer_3.contiguous()
        value_layer_3 = None
        item_10 = (
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
            scale=item_10,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = item_10 = None
        transpose_20 = attn_output_6.transpose(1, 2)
        attn_output_6 = None
        attn_output_7 = transpose_20.contiguous()
        transpose_20 = None
        context_layer_3 = attn_output_7.reshape((1, 236, 32))
        attn_output_7 = None
        hidden_states_27 = torch._C._nn.linear(
            context_layer_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_3 = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_11 = (
            l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_28 = torch.nn.functional.dropout(
            hidden_states_27, item_11, False, False
        )
        hidden_states_27 = item_11 = None
        hidden_states_29 = hidden_states_28 + hidden_states_26
        hidden_states_28 = hidden_states_26 = None
        layer_output_3 = torch.nn.functional.layer_norm(
            hidden_states_29,
            (32,),
            l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_30 = torch._C._nn.linear(
            layer_output_3,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_3 = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_31 = torch._C._nn.gelu(hidden_states_30)
        hidden_states_30 = None
        hidden_states_32 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_31 = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = (None)
        item_12 = (
            l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_33 = torch.nn.functional.dropout(
            hidden_states_32, item_12, False, False
        )
        hidden_states_32 = item_12 = None
        hidden_states_34 = hidden_states_33 + hidden_states_29
        hidden_states_33 = hidden_states_29 = None
        getitem_11 = scale_pos_embed_1[3]
        scale_pos_embed_1 = None
        hidden_states_35 = hidden_states_34 + getitem_11
        hidden_states_34 = getitem_11 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            hidden_states_35,
            (32,),
            l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_bias_ = (None)
        view_15 = linear_24.view(1, -1, 4, 8)
        linear_24 = None
        key_layer_4 = view_15.transpose(1, 2)
        view_15 = None
        linear_25 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_ = (None)
        view_16 = linear_25.view(1, -1, 4, 8)
        linear_25 = None
        value_layer_4 = view_16.transpose(1, 2)
        view_16 = None
        linear_26 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_,
        )
        layer_norm_8 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_ = (None)
        view_17 = linear_26.view(1, -1, 4, 8)
        linear_26 = None
        query_layer_4 = view_17.transpose(1, 2)
        view_17 = None
        query_4 = query_layer_4.contiguous()
        query_layer_4 = None
        key_4 = key_layer_4.contiguous()
        key_layer_4 = None
        value_4 = value_layer_4.contiguous()
        value_layer_4 = None
        item_13 = (
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
            scale=item_13,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = item_13 = None
        transpose_24 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_24.contiguous()
        transpose_24 = None
        context_layer_4 = attn_output_9.reshape((1, 236, 32))
        attn_output_9 = None
        hidden_states_36 = torch._C._nn.linear(
            context_layer_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_4 = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_14 = (
            l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_37 = torch.nn.functional.dropout(
            hidden_states_36, item_14, False, False
        )
        hidden_states_36 = item_14 = None
        hidden_states_38 = hidden_states_37 + hidden_states_35
        hidden_states_37 = hidden_states_35 = None
        layer_output_4 = torch.nn.functional.layer_norm(
            hidden_states_38,
            (32,),
            l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_,
            1e-12,
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_ = (None)
        hidden_states_39 = torch._C._nn.linear(
            layer_output_4,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_4 = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_40 = torch._C._nn.gelu(hidden_states_39)
        hidden_states_39 = None
        hidden_states_41 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_40 = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = (None)
        item_15 = (
            l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_42 = torch.nn.functional.dropout(
            hidden_states_41, item_15, False, False
        )
        hidden_states_41 = item_15 = None
        hidden_states_43 = hidden_states_42 + hidden_states_38
        hidden_states_42 = hidden_states_38 = None
        sequence_output = torch.nn.functional.layer_norm(
            hidden_states_43,
            (32,),
            l_self_modules_layernorm_parameters_weight_,
            l_self_modules_layernorm_parameters_bias_,
            1e-12,
        )
        hidden_states_43 = (
            l_self_modules_layernorm_parameters_weight_
        ) = l_self_modules_layernorm_parameters_bias_ = None
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
