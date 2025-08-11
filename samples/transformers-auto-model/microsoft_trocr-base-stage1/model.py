import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_pixel_values_: torch.Tensor,
        L_self_modules_encoder_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_embeddings_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_embeddings_parameters_position_embeddings_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_embeddings_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_scaling: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_eps: torch.Tensor,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_pooler_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_pooler_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_decoder_input_ids_: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_norm_type: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_embed_scale: torch.Tensor,
        s38: torch.SymInt,
        L_self_modules_decoder_modules_model_modules_decoder_modules_embed_positions_weights: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_embed_positions_buffers_float_tensor_: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_activation_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_activation_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_activation_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_activation_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_activation_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_activation_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_activation_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_activation_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_activation_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_activation_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_activation_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_scaling: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_activation_dropout: torch.Tensor,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_eps: torch.Tensor,
        L_self_modules_decoder_modules_output_projection_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_pixel_values_ = L_pixel_values_
        l_self_modules_encoder_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_ = L_self_modules_encoder_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_
        l_self_modules_encoder_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_ = L_self_modules_encoder_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_
        l_self_modules_encoder_modules_embeddings_parameters_cls_token_ = (
            L_self_modules_encoder_modules_embeddings_parameters_cls_token_
        )
        l_self_modules_encoder_modules_embeddings_parameters_position_embeddings_ = (
            L_self_modules_encoder_modules_embeddings_parameters_position_embeddings_
        )
        l_self_modules_encoder_modules_embeddings_modules_dropout_p = (
            L_self_modules_encoder_modules_embeddings_modules_dropout_p
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_scaling = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_scaling
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_eps = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_eps
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p = L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p
        l_self_modules_encoder_modules_layernorm_parameters_weight_ = (
            L_self_modules_encoder_modules_layernorm_parameters_weight_
        )
        l_self_modules_encoder_modules_layernorm_parameters_bias_ = (
            L_self_modules_encoder_modules_layernorm_parameters_bias_
        )
        l_self_modules_encoder_modules_layernorm_eps = (
            L_self_modules_encoder_modules_layernorm_eps
        )
        l_self_modules_encoder_modules_pooler_modules_dense_parameters_weight_ = (
            L_self_modules_encoder_modules_pooler_modules_dense_parameters_weight_
        )
        l_self_modules_encoder_modules_pooler_modules_dense_parameters_bias_ = (
            L_self_modules_encoder_modules_pooler_modules_dense_parameters_bias_
        )
        l_decoder_input_ids_ = L_decoder_input_ids_
        l_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_norm_type = L_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_norm_type
        l_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_embed_scale = L_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_embed_scale
        l_self_modules_decoder_modules_model_modules_decoder_modules_embed_positions_weights = L_self_modules_decoder_modules_model_modules_decoder_modules_embed_positions_weights
        l_self_modules_decoder_modules_model_modules_decoder_modules_embed_positions_buffers_float_tensor_ = L_self_modules_decoder_modules_model_modules_decoder_modules_embed_positions_buffers_float_tensor_
        l_self_modules_decoder_modules_model_modules_decoder_dropout = (
            L_self_modules_decoder_modules_model_modules_decoder_dropout
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_activation_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_activation_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_activation_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_activation_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_activation_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_activation_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_activation_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_activation_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_activation_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_activation_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_activation_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_activation_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_activation_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_activation_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_activation_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_activation_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_activation_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_activation_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_activation_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_activation_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_activation_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_activation_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_scaling = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_scaling
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_eps
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_activation_dropout = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_activation_dropout
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_eps = L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_eps
        l_self_modules_decoder_modules_output_projection_parameters_weight_ = (
            L_self_modules_decoder_modules_output_projection_parameters_weight_
        )
        conv2d = torch.conv2d(
            l_pixel_values_,
            l_self_modules_encoder_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_,
            l_self_modules_encoder_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        l_pixel_values_ = l_self_modules_encoder_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_ = l_self_modules_encoder_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_ = (None)
        flatten = conv2d.flatten(2)
        conv2d = None
        embeddings = flatten.transpose(1, 2)
        flatten = None
        cls_tokens = (
            l_self_modules_encoder_modules_embeddings_parameters_cls_token_.expand(
                1, -1, -1
            )
        )
        l_self_modules_encoder_modules_embeddings_parameters_cls_token_ = None
        embeddings_1 = torch.cat((cls_tokens, embeddings), dim=1)
        cls_tokens = embeddings = None
        embeddings_2 = (
            embeddings_1
            + l_self_modules_encoder_modules_embeddings_parameters_position_embeddings_
        )
        embeddings_1 = (
            l_self_modules_encoder_modules_embeddings_parameters_position_embeddings_
        ) = None
        item = l_self_modules_encoder_modules_embeddings_modules_dropout_p.item()
        l_self_modules_encoder_modules_embeddings_modules_dropout_p = None
        embeddings_3 = torch.nn.functional.dropout(embeddings_2, item, False, False)
        embeddings_2 = item = None
        item_1 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_eps = (
            None
        )
        layer_norm = torch.nn.functional.layer_norm(
            embeddings_3,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_,
            item_1,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_ = (item_1) = (
            None
        )
        linear = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view = linear.view(1, -1, 12, 64)
        linear = None
        key_layer = view.transpose(1, 2)
        view = None
        linear_1 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_ = (
            None
        )
        view_1 = linear_1.view(1, -1, 12, 64)
        linear_1 = None
        value_layer = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_,
            None,
        )
        layer_norm = l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_ = (None)
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_scaling = (
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer = l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_3 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_1 = torch.nn.functional.dropout(
            hidden_states, item_3, False, False
        )
        hidden_states = item_3 = None
        hidden_states_2 = hidden_states_1 + embeddings_3
        hidden_states_1 = embeddings_3 = None
        item_4 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_eps = (
            None
        )
        layer_output = torch.nn.functional.layer_norm(
            hidden_states_2,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_,
            item_4,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_ = (item_4) = (
            None
        )
        hidden_states_3 = torch._C._nn.linear(
            layer_output,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output = l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_4 = torch._C._nn.gelu(hidden_states_3)
        hidden_states_3 = None
        hidden_states_5 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_4 = l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_ = (None)
        item_5 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, item_5, False, False
        )
        hidden_states_5 = item_5 = None
        hidden_states_7 = hidden_states_6 + hidden_states_2
        hidden_states_6 = hidden_states_2 = None
        item_6 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_eps = (
            None
        )
        layer_norm_2 = torch.nn.functional.layer_norm(
            hidden_states_7,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_,
            item_6,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_ = (item_6) = (
            None
        )
        linear_6 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_3 = linear_6.view(1, -1, 12, 64)
        linear_6 = None
        key_layer_1 = view_3.transpose(1, 2)
        view_3 = None
        linear_7 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_ = (
            None
        )
        view_4 = linear_7.view(1, -1, 12, 64)
        linear_7 = None
        value_layer_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_8 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_,
            None,
        )
        layer_norm_2 = l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_ = (None)
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_scaling = (
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_1 = l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_8 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_9 = torch.nn.functional.dropout(
            hidden_states_8, item_8, False, False
        )
        hidden_states_8 = item_8 = None
        hidden_states_10 = hidden_states_9 + hidden_states_7
        hidden_states_9 = hidden_states_7 = None
        item_9 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_eps = (
            None
        )
        layer_output_1 = torch.nn.functional.layer_norm(
            hidden_states_10,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_,
            item_9,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_ = (item_9) = (
            None
        )
        hidden_states_11 = torch._C._nn.linear(
            layer_output_1,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_1 = l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_12 = torch._C._nn.gelu(hidden_states_11)
        hidden_states_11 = None
        hidden_states_13 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_12 = l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_ = (None)
        item_10 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_14 = torch.nn.functional.dropout(
            hidden_states_13, item_10, False, False
        )
        hidden_states_13 = item_10 = None
        hidden_states_15 = hidden_states_14 + hidden_states_10
        hidden_states_14 = hidden_states_10 = None
        item_11 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_eps = (
            None
        )
        layer_norm_4 = torch.nn.functional.layer_norm(
            hidden_states_15,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_,
            item_11,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_ = (item_11) = (
            None
        )
        linear_12 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_6 = linear_12.view(1, -1, 12, 64)
        linear_12 = None
        key_layer_2 = view_6.transpose(1, 2)
        view_6 = None
        linear_13 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_ = (
            None
        )
        view_7 = linear_13.view(1, -1, 12, 64)
        linear_13 = None
        value_layer_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_14 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_,
            None,
        )
        layer_norm_4 = l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_ = (None)
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_scaling = (
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_13 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_17 = torch.nn.functional.dropout(
            hidden_states_16, item_13, False, False
        )
        hidden_states_16 = item_13 = None
        hidden_states_18 = hidden_states_17 + hidden_states_15
        hidden_states_17 = hidden_states_15 = None
        item_14 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_eps = (
            None
        )
        layer_output_2 = torch.nn.functional.layer_norm(
            hidden_states_18,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_,
            item_14,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_ = (item_14) = (
            None
        )
        hidden_states_19 = torch._C._nn.linear(
            layer_output_2,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_2 = l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_20 = torch._C._nn.gelu(hidden_states_19)
        hidden_states_19 = None
        hidden_states_21 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_20 = l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_ = (None)
        item_15 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, item_15, False, False
        )
        hidden_states_21 = item_15 = None
        hidden_states_23 = hidden_states_22 + hidden_states_18
        hidden_states_22 = hidden_states_18 = None
        item_16 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_eps = (
            None
        )
        layer_norm_6 = torch.nn.functional.layer_norm(
            hidden_states_23,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_,
            item_16,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_ = (item_16) = (
            None
        )
        linear_18 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_9 = linear_18.view(1, -1, 12, 64)
        linear_18 = None
        key_layer_3 = view_9.transpose(1, 2)
        view_9 = None
        linear_19 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_ = (
            None
        )
        view_10 = linear_19.view(1, -1, 12, 64)
        linear_19 = None
        value_layer_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_20 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_,
            None,
        )
        layer_norm_6 = l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_ = (None)
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_scaling = (
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_3 = l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_18 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_25 = torch.nn.functional.dropout(
            hidden_states_24, item_18, False, False
        )
        hidden_states_24 = item_18 = None
        hidden_states_26 = hidden_states_25 + hidden_states_23
        hidden_states_25 = hidden_states_23 = None
        item_19 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_eps = (
            None
        )
        layer_output_3 = torch.nn.functional.layer_norm(
            hidden_states_26,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_,
            item_19,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_ = (item_19) = (
            None
        )
        hidden_states_27 = torch._C._nn.linear(
            layer_output_3,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_3 = l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_28 = torch._C._nn.gelu(hidden_states_27)
        hidden_states_27 = None
        hidden_states_29 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_28 = l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_ = (None)
        item_20 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_30 = torch.nn.functional.dropout(
            hidden_states_29, item_20, False, False
        )
        hidden_states_29 = item_20 = None
        hidden_states_31 = hidden_states_30 + hidden_states_26
        hidden_states_30 = hidden_states_26 = None
        item_21 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_eps = (
            None
        )
        layer_norm_8 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_,
            item_21,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_ = (item_21) = (
            None
        )
        linear_24 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_12 = linear_24.view(1, -1, 12, 64)
        linear_24 = None
        key_layer_4 = view_12.transpose(1, 2)
        view_12 = None
        linear_25 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_ = (
            None
        )
        view_13 = linear_25.view(1, -1, 12, 64)
        linear_25 = None
        value_layer_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_26 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_,
            None,
        )
        layer_norm_8 = l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_ = (None)
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_scaling = (
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_4 = l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_23 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_33 = torch.nn.functional.dropout(
            hidden_states_32, item_23, False, False
        )
        hidden_states_32 = item_23 = None
        hidden_states_34 = hidden_states_33 + hidden_states_31
        hidden_states_33 = hidden_states_31 = None
        item_24 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_eps = (
            None
        )
        layer_output_4 = torch.nn.functional.layer_norm(
            hidden_states_34,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_,
            item_24,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_ = (item_24) = (
            None
        )
        hidden_states_35 = torch._C._nn.linear(
            layer_output_4,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_4 = l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_36 = torch._C._nn.gelu(hidden_states_35)
        hidden_states_35 = None
        hidden_states_37 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_36 = l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_ = (None)
        item_25 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_38 = torch.nn.functional.dropout(
            hidden_states_37, item_25, False, False
        )
        hidden_states_37 = item_25 = None
        hidden_states_39 = hidden_states_38 + hidden_states_34
        hidden_states_38 = hidden_states_34 = None
        item_26 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_eps = (
            None
        )
        layer_norm_10 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_,
            item_26,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_ = (item_26) = (
            None
        )
        linear_30 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_15 = linear_30.view(1, -1, 12, 64)
        linear_30 = None
        key_layer_5 = view_15.transpose(1, 2)
        view_15 = None
        linear_31 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_ = (
            None
        )
        view_16 = linear_31.view(1, -1, 12, 64)
        linear_31 = None
        value_layer_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_32 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_,
            None,
        )
        layer_norm_10 = l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_ = (None)
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_scaling = (
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_5 = l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_28 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_41 = torch.nn.functional.dropout(
            hidden_states_40, item_28, False, False
        )
        hidden_states_40 = item_28 = None
        hidden_states_42 = hidden_states_41 + hidden_states_39
        hidden_states_41 = hidden_states_39 = None
        item_29 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_eps = (
            None
        )
        layer_output_5 = torch.nn.functional.layer_norm(
            hidden_states_42,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_,
            item_29,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_ = (item_29) = (
            None
        )
        hidden_states_43 = torch._C._nn.linear(
            layer_output_5,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_5 = l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_44 = torch._C._nn.gelu(hidden_states_43)
        hidden_states_43 = None
        hidden_states_45 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_44 = l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_ = (None)
        item_30 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_46 = torch.nn.functional.dropout(
            hidden_states_45, item_30, False, False
        )
        hidden_states_45 = item_30 = None
        hidden_states_47 = hidden_states_46 + hidden_states_42
        hidden_states_46 = hidden_states_42 = None
        item_31 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_eps = (
            None
        )
        layer_norm_12 = torch.nn.functional.layer_norm(
            hidden_states_47,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_,
            item_31,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_ = (item_31) = (
            None
        )
        linear_36 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_18 = linear_36.view(1, -1, 12, 64)
        linear_36 = None
        key_layer_6 = view_18.transpose(1, 2)
        view_18 = None
        linear_37 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_ = (
            None
        )
        view_19 = linear_37.view(1, -1, 12, 64)
        linear_37 = None
        value_layer_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_38 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_,
            None,
        )
        layer_norm_12 = l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_ = (None)
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_scaling = (
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_6 = l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_33 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_49 = torch.nn.functional.dropout(
            hidden_states_48, item_33, False, False
        )
        hidden_states_48 = item_33 = None
        hidden_states_50 = hidden_states_49 + hidden_states_47
        hidden_states_49 = hidden_states_47 = None
        item_34 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_eps = (
            None
        )
        layer_output_6 = torch.nn.functional.layer_norm(
            hidden_states_50,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_,
            item_34,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_ = (item_34) = (
            None
        )
        hidden_states_51 = torch._C._nn.linear(
            layer_output_6,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_6 = l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_52 = torch._C._nn.gelu(hidden_states_51)
        hidden_states_51 = None
        hidden_states_53 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_52 = l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_ = (None)
        item_35 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_54 = torch.nn.functional.dropout(
            hidden_states_53, item_35, False, False
        )
        hidden_states_53 = item_35 = None
        hidden_states_55 = hidden_states_54 + hidden_states_50
        hidden_states_54 = hidden_states_50 = None
        item_36 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_eps = (
            None
        )
        layer_norm_14 = torch.nn.functional.layer_norm(
            hidden_states_55,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_,
            item_36,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_ = (item_36) = (
            None
        )
        linear_42 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_21 = linear_42.view(1, -1, 12, 64)
        linear_42 = None
        key_layer_7 = view_21.transpose(1, 2)
        view_21 = None
        linear_43 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_ = (
            None
        )
        view_22 = linear_43.view(1, -1, 12, 64)
        linear_43 = None
        value_layer_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_44 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_,
            None,
        )
        layer_norm_14 = l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_ = (None)
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_scaling = (
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_7 = l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_38 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_57 = torch.nn.functional.dropout(
            hidden_states_56, item_38, False, False
        )
        hidden_states_56 = item_38 = None
        hidden_states_58 = hidden_states_57 + hidden_states_55
        hidden_states_57 = hidden_states_55 = None
        item_39 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_eps = (
            None
        )
        layer_output_7 = torch.nn.functional.layer_norm(
            hidden_states_58,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_,
            item_39,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_ = (item_39) = (
            None
        )
        hidden_states_59 = torch._C._nn.linear(
            layer_output_7,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_7 = l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_60 = torch._C._nn.gelu(hidden_states_59)
        hidden_states_59 = None
        hidden_states_61 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_60 = l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_ = (None)
        item_40 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_62 = torch.nn.functional.dropout(
            hidden_states_61, item_40, False, False
        )
        hidden_states_61 = item_40 = None
        hidden_states_63 = hidden_states_62 + hidden_states_58
        hidden_states_62 = hidden_states_58 = None
        item_41 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_eps = (
            None
        )
        layer_norm_16 = torch.nn.functional.layer_norm(
            hidden_states_63,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_,
            item_41,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_ = (item_41) = (
            None
        )
        linear_48 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_24 = linear_48.view(1, -1, 12, 64)
        linear_48 = None
        key_layer_8 = view_24.transpose(1, 2)
        view_24 = None
        linear_49 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_ = (
            None
        )
        view_25 = linear_49.view(1, -1, 12, 64)
        linear_49 = None
        value_layer_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_50 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_,
            None,
        )
        layer_norm_16 = l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_ = (None)
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_scaling = (
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_8 = l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_43 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_65 = torch.nn.functional.dropout(
            hidden_states_64, item_43, False, False
        )
        hidden_states_64 = item_43 = None
        hidden_states_66 = hidden_states_65 + hidden_states_63
        hidden_states_65 = hidden_states_63 = None
        item_44 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_eps = (
            None
        )
        layer_output_8 = torch.nn.functional.layer_norm(
            hidden_states_66,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_,
            item_44,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_ = (item_44) = (
            None
        )
        hidden_states_67 = torch._C._nn.linear(
            layer_output_8,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_8 = l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_68 = torch._C._nn.gelu(hidden_states_67)
        hidden_states_67 = None
        hidden_states_69 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_68 = l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_ = (None)
        item_45 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, item_45, False, False
        )
        hidden_states_69 = item_45 = None
        hidden_states_71 = hidden_states_70 + hidden_states_66
        hidden_states_70 = hidden_states_66 = None
        item_46 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_eps = (
            None
        )
        layer_norm_18 = torch.nn.functional.layer_norm(
            hidden_states_71,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_,
            item_46,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_ = (item_46) = (
            None
        )
        linear_54 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_27 = linear_54.view(1, -1, 12, 64)
        linear_54 = None
        key_layer_9 = view_27.transpose(1, 2)
        view_27 = None
        linear_55 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_ = (
            None
        )
        view_28 = linear_55.view(1, -1, 12, 64)
        linear_55 = None
        value_layer_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_56 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_,
            None,
        )
        layer_norm_18 = l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_ = (None)
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_scaling = (
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_9 = l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_48 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_73 = torch.nn.functional.dropout(
            hidden_states_72, item_48, False, False
        )
        hidden_states_72 = item_48 = None
        hidden_states_74 = hidden_states_73 + hidden_states_71
        hidden_states_73 = hidden_states_71 = None
        item_49 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_eps = (
            None
        )
        layer_output_9 = torch.nn.functional.layer_norm(
            hidden_states_74,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_,
            item_49,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_ = (item_49) = (
            None
        )
        hidden_states_75 = torch._C._nn.linear(
            layer_output_9,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_9 = l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_76 = torch._C._nn.gelu(hidden_states_75)
        hidden_states_75 = None
        hidden_states_77 = torch._C._nn.linear(
            hidden_states_76,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_76 = l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_ = (None)
        item_50 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_78 = torch.nn.functional.dropout(
            hidden_states_77, item_50, False, False
        )
        hidden_states_77 = item_50 = None
        hidden_states_79 = hidden_states_78 + hidden_states_74
        hidden_states_78 = hidden_states_74 = None
        item_51 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_eps = (
            None
        )
        layer_norm_20 = torch.nn.functional.layer_norm(
            hidden_states_79,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_,
            item_51,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_ = (item_51) = (
            None
        )
        linear_60 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_30 = linear_60.view(1, -1, 12, 64)
        linear_60 = None
        key_layer_10 = view_30.transpose(1, 2)
        view_30 = None
        linear_61 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_ = (
            None
        )
        view_31 = linear_61.view(1, -1, 12, 64)
        linear_61 = None
        value_layer_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_62 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_,
            None,
        )
        layer_norm_20 = l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_ = (None)
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_scaling = (
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_10 = l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_53 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_81 = torch.nn.functional.dropout(
            hidden_states_80, item_53, False, False
        )
        hidden_states_80 = item_53 = None
        hidden_states_82 = hidden_states_81 + hidden_states_79
        hidden_states_81 = hidden_states_79 = None
        item_54 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_eps = (
            None
        )
        layer_output_10 = torch.nn.functional.layer_norm(
            hidden_states_82,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_,
            item_54,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_ = (item_54) = (
            None
        )
        hidden_states_83 = torch._C._nn.linear(
            layer_output_10,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_10 = l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_84 = torch._C._nn.gelu(hidden_states_83)
        hidden_states_83 = None
        hidden_states_85 = torch._C._nn.linear(
            hidden_states_84,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_84 = l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_ = (None)
        item_55 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_86 = torch.nn.functional.dropout(
            hidden_states_85, item_55, False, False
        )
        hidden_states_85 = item_55 = None
        hidden_states_87 = hidden_states_86 + hidden_states_82
        hidden_states_86 = hidden_states_82 = None
        item_56 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_eps = (
            None
        )
        layer_norm_22 = torch.nn.functional.layer_norm(
            hidden_states_87,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_,
            item_56,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_ = (item_56) = (
            None
        )
        linear_66 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_ = (
            None
        )
        view_33 = linear_66.view(1, -1, 12, 64)
        linear_66 = None
        key_layer_11 = view_33.transpose(1, 2)
        view_33 = None
        linear_67 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_,
            None,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_ = (
            None
        )
        view_34 = linear_67.view(1, -1, 12, 64)
        linear_67 = None
        value_layer_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_68 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_,
            None,
        )
        layer_norm_22 = l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_ = (None)
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_scaling.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_scaling = (
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
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        item_58 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_89 = torch.nn.functional.dropout(
            hidden_states_88, item_58, False, False
        )
        hidden_states_88 = item_58 = None
        hidden_states_90 = hidden_states_89 + hidden_states_87
        hidden_states_89 = hidden_states_87 = None
        item_59 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_eps.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_eps = (
            None
        )
        layer_output_11 = torch.nn.functional.layer_norm(
            hidden_states_90,
            (768,),
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_,
            item_59,
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_ = (item_59) = (
            None
        )
        hidden_states_91 = torch._C._nn.linear(
            layer_output_11,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_,
        )
        layer_output_11 = l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_ = (None)
        hidden_states_92 = torch._C._nn.gelu(hidden_states_91)
        hidden_states_91 = None
        hidden_states_93 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_ = (None)
        item_60 = (
            l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dropout_p = (
            None
        )
        hidden_states_94 = torch.nn.functional.dropout(
            hidden_states_93, item_60, False, False
        )
        hidden_states_93 = item_60 = None
        hidden_states_95 = hidden_states_94 + hidden_states_90
        hidden_states_94 = hidden_states_90 = None
        item_61 = l_self_modules_encoder_modules_layernorm_eps.item()
        l_self_modules_encoder_modules_layernorm_eps = None
        sequence_output = torch.nn.functional.layer_norm(
            hidden_states_95,
            (768,),
            l_self_modules_encoder_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_layernorm_parameters_bias_,
            item_61,
        )
        hidden_states_95 = (
            l_self_modules_encoder_modules_layernorm_parameters_weight_
        ) = l_self_modules_encoder_modules_layernorm_parameters_bias_ = item_61 = None
        first_token_tensor = sequence_output[(slice(None, None, None), 0)]
        pooled_output = torch._C._nn.linear(
            first_token_tensor,
            l_self_modules_encoder_modules_pooler_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_pooler_modules_dense_parameters_bias_,
        )
        first_token_tensor = (
            l_self_modules_encoder_modules_pooler_modules_dense_parameters_weight_
        ) = l_self_modules_encoder_modules_pooler_modules_dense_parameters_bias_ = None
        pooled_output_1 = torch.tanh(pooled_output)
        pooled_output = pooled_output_1 = None
        input_ids = l_decoder_input_ids_.view(-1, 1)
        l_decoder_input_ids_ = None
        item_62 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_norm_type.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_norm_type = (
            None
        )
        embedding = torch.nn.functional.embedding(
            input_ids,
            l_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_,
            1,
            None,
            item_62,
            False,
            False,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_ = (
            item_62
        ) = None
        item_63 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_embed_scale.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_embed_scale = (
            None
        )
        inputs_embeds = embedding * item_63
        embedding = item_63 = None
        ne_3 = input_ids.ne(1)
        input_ids = None
        mask = ne_3.int()
        ne_3 = None
        cumsum = torch.cumsum(mask, dim=1)
        type_as = cumsum.type_as(mask)
        cumsum = None
        add_25 = type_as + 0
        type_as = None
        incremental_indices = add_25 * mask
        add_25 = mask = None
        long = incremental_indices.long()
        incremental_indices = None
        add_26 = long + 1
        long = None
        position_ids = add_26.to(device(type="cuda", index=0))
        add_26 = None
        to_1 = l_self_modules_decoder_modules_model_modules_decoder_modules_embed_positions_weights.to(
            l_self_modules_decoder_modules_model_modules_decoder_modules_embed_positions_buffers_float_tensor_
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_embed_positions_weights = l_self_modules_decoder_modules_model_modules_decoder_modules_embed_positions_buffers_float_tensor_ = (None)
        view_37 = position_ids.view(-1)
        position_ids = None
        index_select = to_1.index_select(0, view_37)
        view_37 = None
        view_38 = index_select.view(1, 1, -1)
        index_select = None
        x = view_38.detach()
        view_38 = None
        hidden_states_96 = inputs_embeds + x
        inputs_embeds = x = None
        item_64 = l_self_modules_decoder_modules_model_modules_decoder_dropout.item()
        l_self_modules_decoder_modules_model_modules_decoder_dropout = None
        hidden_states_97 = torch.nn.functional.dropout(
            hidden_states_96, p=item_64, training=False
        )
        hidden_states_96 = item_64 = None
        linear_73 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        item_65 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_scaling = (
            None
        )
        query_states = linear_73 * item_65
        linear_73 = item_65 = None
        key_states = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_39 = key_states.view(1, -1, 16, 64)
        key_states = None
        key_states_1 = view_39.transpose(1, 2)
        view_39 = None
        view_40 = value_states.view(1, -1, 16, 64)
        value_states = None
        value_states_1 = view_40.transpose(1, 2)
        view_40 = None
        view_41 = query_states.view(1, 1, 16, 64)
        query_states = None
        query_states_1 = view_41.transpose(1, 2)
        view_41 = None
        query_states_2 = query_states_1.reshape(16, -1, 64)
        query_states_1 = None
        key_states_2 = key_states_1.reshape(16, -1, 64)
        key_states_1 = None
        value_states_2 = value_states_1.reshape(16, -1, 64)
        value_states_1 = None
        transpose_52 = key_states_2.transpose(1, 2)
        key_states_2 = None
        attn_weights = torch.bmm(query_states_2, transpose_52)
        query_states_2 = transpose_52 = None
        attn_weights_1 = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = None
        item_66 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_dropout = (
            None
        )
        attn_probs = torch.nn.functional.dropout(
            attn_weights_1, p=item_66, training=False
        )
        attn_weights_1 = item_66 = None
        attn_output_24 = torch.bmm(attn_probs, value_states_2)
        attn_probs = value_states_2 = None
        attn_output_25 = attn_output_24.view(1, 16, 1, 64)
        attn_output_24 = None
        attn_output_26 = attn_output_25.transpose(1, 2)
        attn_output_25 = None
        attn_output_27 = attn_output_26.reshape(1, 1, 1024)
        attn_output_26 = None
        attn_output_28 = torch._C._nn.linear(
            attn_output_27,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_27 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        item_67 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_dropout = (
            None
        )
        hidden_states_98 = torch.nn.functional.dropout(
            attn_output_28, p=item_67, training=False
        )
        attn_output_28 = None
        hidden_states_99 = hidden_states_97 + hidden_states_98
        hidden_states_97 = hidden_states_98 = None
        item_68 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_eps = (
            None
        )
        hidden_states_100 = torch.nn.functional.layer_norm(
            hidden_states_99,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            item_68,
        )
        hidden_states_99 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (item_68) = (
            None
        )
        linear_77 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        item_69 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_scaling = (
            None
        )
        query_states_3 = linear_77 * item_69
        linear_77 = item_69 = None
        key_states_3 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_3 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_43 = key_states_3.view(1, -1, 16, 64)
        key_states_3 = None
        key_states_4 = view_43.transpose(1, 2)
        view_43 = None
        view_44 = value_states_3.view(1, -1, 16, 64)
        value_states_3 = None
        value_states_4 = view_44.transpose(1, 2)
        view_44 = None
        view_45 = query_states_3.view(1, 1, 16, 64)
        query_states_3 = None
        query_states_4 = view_45.transpose(1, 2)
        view_45 = None
        query_states_5 = query_states_4.reshape(16, -1, 64)
        query_states_4 = None
        key_states_5 = key_states_4.reshape(16, -1, 64)
        key_states_4 = None
        value_states_5 = value_states_4.reshape(16, -1, 64)
        value_states_4 = None
        transpose_57 = key_states_5.transpose(1, 2)
        key_states_5 = None
        attn_weights_2 = torch.bmm(query_states_5, transpose_57)
        query_states_5 = transpose_57 = None
        attn_weights_3 = torch.nn.functional.softmax(attn_weights_2, dim=-1)
        attn_weights_2 = None
        item_70 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_dropout = (
            None
        )
        attn_probs_1 = torch.nn.functional.dropout(
            attn_weights_3, p=item_70, training=False
        )
        attn_weights_3 = item_70 = None
        attn_output_29 = torch.bmm(attn_probs_1, value_states_5)
        attn_probs_1 = value_states_5 = None
        attn_output_30 = attn_output_29.view(1, 16, 1, 64)
        attn_output_29 = None
        attn_output_31 = attn_output_30.transpose(1, 2)
        attn_output_30 = None
        attn_output_32 = attn_output_31.reshape(1, 1, 1024)
        attn_output_31 = None
        attn_output_33 = torch._C._nn.linear(
            attn_output_32,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_32 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_101 = torch.nn.functional.dropout(
            attn_output_33, p=item_67, training=False
        )
        attn_output_33 = None
        hidden_states_102 = hidden_states_100 + hidden_states_101
        hidden_states_100 = hidden_states_101 = None
        item_71 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_eps = (
            None
        )
        hidden_states_103 = torch.nn.functional.layer_norm(
            hidden_states_102,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_,
            item_71,
        )
        hidden_states_102 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = (item_71) = (
            None
        )
        linear_81 = torch._C._nn.linear(
            hidden_states_103,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_ = (None)
        hidden_states_104 = torch.nn.functional.relu(linear_81, inplace=False)
        linear_81 = None
        item_72 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_activation_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_activation_dropout = (
            None
        )
        hidden_states_105 = torch.nn.functional.dropout(
            hidden_states_104, p=item_72, training=False
        )
        hidden_states_104 = item_72 = None
        hidden_states_106 = torch._C._nn.linear(
            hidden_states_105,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_105 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_ = (None)
        hidden_states_107 = torch.nn.functional.dropout(
            hidden_states_106, p=item_67, training=False
        )
        hidden_states_106 = item_67 = None
        hidden_states_108 = hidden_states_103 + hidden_states_107
        hidden_states_103 = hidden_states_107 = None
        item_73 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_eps = (
            None
        )
        hidden_states_109 = torch.nn.functional.layer_norm(
            hidden_states_108,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            item_73,
        )
        hidden_states_108 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (item_73) = (
            None
        )
        linear_83 = torch._C._nn.linear(
            hidden_states_109,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        item_74 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_scaling = (
            None
        )
        query_states_6 = linear_83 * item_74
        linear_83 = item_74 = None
        key_states_6 = torch._C._nn.linear(
            hidden_states_109,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_6 = torch._C._nn.linear(
            hidden_states_109,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_47 = key_states_6.view(1, -1, 16, 64)
        key_states_6 = None
        key_states_7 = view_47.transpose(1, 2)
        view_47 = None
        view_48 = value_states_6.view(1, -1, 16, 64)
        value_states_6 = None
        value_states_7 = view_48.transpose(1, 2)
        view_48 = None
        view_49 = query_states_6.view(1, 1, 16, 64)
        query_states_6 = None
        query_states_7 = view_49.transpose(1, 2)
        view_49 = None
        query_states_8 = query_states_7.reshape(16, -1, 64)
        query_states_7 = None
        key_states_8 = key_states_7.reshape(16, -1, 64)
        key_states_7 = None
        value_states_8 = value_states_7.reshape(16, -1, 64)
        value_states_7 = None
        transpose_62 = key_states_8.transpose(1, 2)
        key_states_8 = None
        attn_weights_4 = torch.bmm(query_states_8, transpose_62)
        query_states_8 = transpose_62 = None
        attn_weights_5 = torch.nn.functional.softmax(attn_weights_4, dim=-1)
        attn_weights_4 = None
        item_75 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_dropout = (
            None
        )
        attn_probs_2 = torch.nn.functional.dropout(
            attn_weights_5, p=item_75, training=False
        )
        attn_weights_5 = item_75 = None
        attn_output_34 = torch.bmm(attn_probs_2, value_states_8)
        attn_probs_2 = value_states_8 = None
        attn_output_35 = attn_output_34.view(1, 16, 1, 64)
        attn_output_34 = None
        attn_output_36 = attn_output_35.transpose(1, 2)
        attn_output_35 = None
        attn_output_37 = attn_output_36.reshape(1, 1, 1024)
        attn_output_36 = None
        attn_output_38 = torch._C._nn.linear(
            attn_output_37,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_37 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        item_76 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_dropout = (
            None
        )
        hidden_states_110 = torch.nn.functional.dropout(
            attn_output_38, p=item_76, training=False
        )
        attn_output_38 = None
        hidden_states_111 = hidden_states_109 + hidden_states_110
        hidden_states_109 = hidden_states_110 = None
        item_77 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_eps = (
            None
        )
        hidden_states_112 = torch.nn.functional.layer_norm(
            hidden_states_111,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            item_77,
        )
        hidden_states_111 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (item_77) = (
            None
        )
        linear_87 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        item_78 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_scaling = (
            None
        )
        query_states_9 = linear_87 * item_78
        linear_87 = item_78 = None
        key_states_9 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_9 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_51 = key_states_9.view(1, -1, 16, 64)
        key_states_9 = None
        key_states_10 = view_51.transpose(1, 2)
        view_51 = None
        view_52 = value_states_9.view(1, -1, 16, 64)
        value_states_9 = None
        value_states_10 = view_52.transpose(1, 2)
        view_52 = None
        view_53 = query_states_9.view(1, 1, 16, 64)
        query_states_9 = None
        query_states_10 = view_53.transpose(1, 2)
        view_53 = None
        query_states_11 = query_states_10.reshape(16, -1, 64)
        query_states_10 = None
        key_states_11 = key_states_10.reshape(16, -1, 64)
        key_states_10 = None
        value_states_11 = value_states_10.reshape(16, -1, 64)
        value_states_10 = None
        transpose_67 = key_states_11.transpose(1, 2)
        key_states_11 = None
        attn_weights_6 = torch.bmm(query_states_11, transpose_67)
        query_states_11 = transpose_67 = None
        attn_weights_7 = torch.nn.functional.softmax(attn_weights_6, dim=-1)
        attn_weights_6 = None
        item_79 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_dropout = (
            None
        )
        attn_probs_3 = torch.nn.functional.dropout(
            attn_weights_7, p=item_79, training=False
        )
        attn_weights_7 = item_79 = None
        attn_output_39 = torch.bmm(attn_probs_3, value_states_11)
        attn_probs_3 = value_states_11 = None
        attn_output_40 = attn_output_39.view(1, 16, 1, 64)
        attn_output_39 = None
        attn_output_41 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_42 = attn_output_41.reshape(1, 1, 1024)
        attn_output_41 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_42 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_113 = torch.nn.functional.dropout(
            attn_output_43, p=item_76, training=False
        )
        attn_output_43 = None
        hidden_states_114 = hidden_states_112 + hidden_states_113
        hidden_states_112 = hidden_states_113 = None
        item_80 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_eps = (
            None
        )
        hidden_states_115 = torch.nn.functional.layer_norm(
            hidden_states_114,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_,
            item_80,
        )
        hidden_states_114 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = (item_80) = (
            None
        )
        linear_91 = torch._C._nn.linear(
            hidden_states_115,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_ = (None)
        hidden_states_116 = torch.nn.functional.relu(linear_91, inplace=False)
        linear_91 = None
        item_81 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_activation_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_activation_dropout = (
            None
        )
        hidden_states_117 = torch.nn.functional.dropout(
            hidden_states_116, p=item_81, training=False
        )
        hidden_states_116 = item_81 = None
        hidden_states_118 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_117 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_ = (None)
        hidden_states_119 = torch.nn.functional.dropout(
            hidden_states_118, p=item_76, training=False
        )
        hidden_states_118 = item_76 = None
        hidden_states_120 = hidden_states_115 + hidden_states_119
        hidden_states_115 = hidden_states_119 = None
        item_82 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_eps = (
            None
        )
        hidden_states_121 = torch.nn.functional.layer_norm(
            hidden_states_120,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            item_82,
        )
        hidden_states_120 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (item_82) = (
            None
        )
        linear_93 = torch._C._nn.linear(
            hidden_states_121,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        item_83 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_scaling = (
            None
        )
        query_states_12 = linear_93 * item_83
        linear_93 = item_83 = None
        key_states_12 = torch._C._nn.linear(
            hidden_states_121,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_12 = torch._C._nn.linear(
            hidden_states_121,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_55 = key_states_12.view(1, -1, 16, 64)
        key_states_12 = None
        key_states_13 = view_55.transpose(1, 2)
        view_55 = None
        view_56 = value_states_12.view(1, -1, 16, 64)
        value_states_12 = None
        value_states_13 = view_56.transpose(1, 2)
        view_56 = None
        view_57 = query_states_12.view(1, 1, 16, 64)
        query_states_12 = None
        query_states_13 = view_57.transpose(1, 2)
        view_57 = None
        query_states_14 = query_states_13.reshape(16, -1, 64)
        query_states_13 = None
        key_states_14 = key_states_13.reshape(16, -1, 64)
        key_states_13 = None
        value_states_14 = value_states_13.reshape(16, -1, 64)
        value_states_13 = None
        transpose_72 = key_states_14.transpose(1, 2)
        key_states_14 = None
        attn_weights_8 = torch.bmm(query_states_14, transpose_72)
        query_states_14 = transpose_72 = None
        attn_weights_9 = torch.nn.functional.softmax(attn_weights_8, dim=-1)
        attn_weights_8 = None
        item_84 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_dropout = (
            None
        )
        attn_probs_4 = torch.nn.functional.dropout(
            attn_weights_9, p=item_84, training=False
        )
        attn_weights_9 = item_84 = None
        attn_output_44 = torch.bmm(attn_probs_4, value_states_14)
        attn_probs_4 = value_states_14 = None
        attn_output_45 = attn_output_44.view(1, 16, 1, 64)
        attn_output_44 = None
        attn_output_46 = attn_output_45.transpose(1, 2)
        attn_output_45 = None
        attn_output_47 = attn_output_46.reshape(1, 1, 1024)
        attn_output_46 = None
        attn_output_48 = torch._C._nn.linear(
            attn_output_47,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_47 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        item_85 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_dropout = (
            None
        )
        hidden_states_122 = torch.nn.functional.dropout(
            attn_output_48, p=item_85, training=False
        )
        attn_output_48 = None
        hidden_states_123 = hidden_states_121 + hidden_states_122
        hidden_states_121 = hidden_states_122 = None
        item_86 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_eps = (
            None
        )
        hidden_states_124 = torch.nn.functional.layer_norm(
            hidden_states_123,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            item_86,
        )
        hidden_states_123 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (item_86) = (
            None
        )
        linear_97 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        item_87 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_scaling = (
            None
        )
        query_states_15 = linear_97 * item_87
        linear_97 = item_87 = None
        key_states_15 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_15 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_59 = key_states_15.view(1, -1, 16, 64)
        key_states_15 = None
        key_states_16 = view_59.transpose(1, 2)
        view_59 = None
        view_60 = value_states_15.view(1, -1, 16, 64)
        value_states_15 = None
        value_states_16 = view_60.transpose(1, 2)
        view_60 = None
        view_61 = query_states_15.view(1, 1, 16, 64)
        query_states_15 = None
        query_states_16 = view_61.transpose(1, 2)
        view_61 = None
        query_states_17 = query_states_16.reshape(16, -1, 64)
        query_states_16 = None
        key_states_17 = key_states_16.reshape(16, -1, 64)
        key_states_16 = None
        value_states_17 = value_states_16.reshape(16, -1, 64)
        value_states_16 = None
        transpose_77 = key_states_17.transpose(1, 2)
        key_states_17 = None
        attn_weights_10 = torch.bmm(query_states_17, transpose_77)
        query_states_17 = transpose_77 = None
        attn_weights_11 = torch.nn.functional.softmax(attn_weights_10, dim=-1)
        attn_weights_10 = None
        item_88 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_dropout = (
            None
        )
        attn_probs_5 = torch.nn.functional.dropout(
            attn_weights_11, p=item_88, training=False
        )
        attn_weights_11 = item_88 = None
        attn_output_49 = torch.bmm(attn_probs_5, value_states_17)
        attn_probs_5 = value_states_17 = None
        attn_output_50 = attn_output_49.view(1, 16, 1, 64)
        attn_output_49 = None
        attn_output_51 = attn_output_50.transpose(1, 2)
        attn_output_50 = None
        attn_output_52 = attn_output_51.reshape(1, 1, 1024)
        attn_output_51 = None
        attn_output_53 = torch._C._nn.linear(
            attn_output_52,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_52 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_125 = torch.nn.functional.dropout(
            attn_output_53, p=item_85, training=False
        )
        attn_output_53 = None
        hidden_states_126 = hidden_states_124 + hidden_states_125
        hidden_states_124 = hidden_states_125 = None
        item_89 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_eps = (
            None
        )
        hidden_states_127 = torch.nn.functional.layer_norm(
            hidden_states_126,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_,
            item_89,
        )
        hidden_states_126 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_ = (item_89) = (
            None
        )
        linear_101 = torch._C._nn.linear(
            hidden_states_127,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_ = (None)
        hidden_states_128 = torch.nn.functional.relu(linear_101, inplace=False)
        linear_101 = None
        item_90 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_activation_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_activation_dropout = (
            None
        )
        hidden_states_129 = torch.nn.functional.dropout(
            hidden_states_128, p=item_90, training=False
        )
        hidden_states_128 = item_90 = None
        hidden_states_130 = torch._C._nn.linear(
            hidden_states_129,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_,
        )
        hidden_states_129 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_ = (None)
        hidden_states_131 = torch.nn.functional.dropout(
            hidden_states_130, p=item_85, training=False
        )
        hidden_states_130 = item_85 = None
        hidden_states_132 = hidden_states_127 + hidden_states_131
        hidden_states_127 = hidden_states_131 = None
        item_91 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_eps = (
            None
        )
        hidden_states_133 = torch.nn.functional.layer_norm(
            hidden_states_132,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            item_91,
        )
        hidden_states_132 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = (item_91) = (
            None
        )
        linear_103 = torch._C._nn.linear(
            hidden_states_133,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        item_92 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_scaling = (
            None
        )
        query_states_18 = linear_103 * item_92
        linear_103 = item_92 = None
        key_states_18 = torch._C._nn.linear(
            hidden_states_133,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_18 = torch._C._nn.linear(
            hidden_states_133,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_63 = key_states_18.view(1, -1, 16, 64)
        key_states_18 = None
        key_states_19 = view_63.transpose(1, 2)
        view_63 = None
        view_64 = value_states_18.view(1, -1, 16, 64)
        value_states_18 = None
        value_states_19 = view_64.transpose(1, 2)
        view_64 = None
        view_65 = query_states_18.view(1, 1, 16, 64)
        query_states_18 = None
        query_states_19 = view_65.transpose(1, 2)
        view_65 = None
        query_states_20 = query_states_19.reshape(16, -1, 64)
        query_states_19 = None
        key_states_20 = key_states_19.reshape(16, -1, 64)
        key_states_19 = None
        value_states_20 = value_states_19.reshape(16, -1, 64)
        value_states_19 = None
        transpose_82 = key_states_20.transpose(1, 2)
        key_states_20 = None
        attn_weights_12 = torch.bmm(query_states_20, transpose_82)
        query_states_20 = transpose_82 = None
        attn_weights_13 = torch.nn.functional.softmax(attn_weights_12, dim=-1)
        attn_weights_12 = None
        item_93 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_dropout = (
            None
        )
        attn_probs_6 = torch.nn.functional.dropout(
            attn_weights_13, p=item_93, training=False
        )
        attn_weights_13 = item_93 = None
        attn_output_54 = torch.bmm(attn_probs_6, value_states_20)
        attn_probs_6 = value_states_20 = None
        attn_output_55 = attn_output_54.view(1, 16, 1, 64)
        attn_output_54 = None
        attn_output_56 = attn_output_55.transpose(1, 2)
        attn_output_55 = None
        attn_output_57 = attn_output_56.reshape(1, 1, 1024)
        attn_output_56 = None
        attn_output_58 = torch._C._nn.linear(
            attn_output_57,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_57 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        item_94 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_dropout = (
            None
        )
        hidden_states_134 = torch.nn.functional.dropout(
            attn_output_58, p=item_94, training=False
        )
        attn_output_58 = None
        hidden_states_135 = hidden_states_133 + hidden_states_134
        hidden_states_133 = hidden_states_134 = None
        item_95 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_eps = (
            None
        )
        hidden_states_136 = torch.nn.functional.layer_norm(
            hidden_states_135,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            item_95,
        )
        hidden_states_135 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (item_95) = (
            None
        )
        linear_107 = torch._C._nn.linear(
            hidden_states_136,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        item_96 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_scaling = (
            None
        )
        query_states_21 = linear_107 * item_96
        linear_107 = item_96 = None
        key_states_21 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_21 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_67 = key_states_21.view(1, -1, 16, 64)
        key_states_21 = None
        key_states_22 = view_67.transpose(1, 2)
        view_67 = None
        view_68 = value_states_21.view(1, -1, 16, 64)
        value_states_21 = None
        value_states_22 = view_68.transpose(1, 2)
        view_68 = None
        view_69 = query_states_21.view(1, 1, 16, 64)
        query_states_21 = None
        query_states_22 = view_69.transpose(1, 2)
        view_69 = None
        query_states_23 = query_states_22.reshape(16, -1, 64)
        query_states_22 = None
        key_states_23 = key_states_22.reshape(16, -1, 64)
        key_states_22 = None
        value_states_23 = value_states_22.reshape(16, -1, 64)
        value_states_22 = None
        transpose_87 = key_states_23.transpose(1, 2)
        key_states_23 = None
        attn_weights_14 = torch.bmm(query_states_23, transpose_87)
        query_states_23 = transpose_87 = None
        attn_weights_15 = torch.nn.functional.softmax(attn_weights_14, dim=-1)
        attn_weights_14 = None
        item_97 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_dropout = (
            None
        )
        attn_probs_7 = torch.nn.functional.dropout(
            attn_weights_15, p=item_97, training=False
        )
        attn_weights_15 = item_97 = None
        attn_output_59 = torch.bmm(attn_probs_7, value_states_23)
        attn_probs_7 = value_states_23 = None
        attn_output_60 = attn_output_59.view(1, 16, 1, 64)
        attn_output_59 = None
        attn_output_61 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_62 = attn_output_61.reshape(1, 1, 1024)
        attn_output_61 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_62 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_137 = torch.nn.functional.dropout(
            attn_output_63, p=item_94, training=False
        )
        attn_output_63 = None
        hidden_states_138 = hidden_states_136 + hidden_states_137
        hidden_states_136 = hidden_states_137 = None
        item_98 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_eps = (
            None
        )
        hidden_states_139 = torch.nn.functional.layer_norm(
            hidden_states_138,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_,
            item_98,
        )
        hidden_states_138 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_ = (item_98) = (
            None
        )
        linear_111 = torch._C._nn.linear(
            hidden_states_139,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_ = (None)
        hidden_states_140 = torch.nn.functional.relu(linear_111, inplace=False)
        linear_111 = None
        item_99 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_activation_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_activation_dropout = (
            None
        )
        hidden_states_141 = torch.nn.functional.dropout(
            hidden_states_140, p=item_99, training=False
        )
        hidden_states_140 = item_99 = None
        hidden_states_142 = torch._C._nn.linear(
            hidden_states_141,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_,
        )
        hidden_states_141 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_ = (None)
        hidden_states_143 = torch.nn.functional.dropout(
            hidden_states_142, p=item_94, training=False
        )
        hidden_states_142 = item_94 = None
        hidden_states_144 = hidden_states_139 + hidden_states_143
        hidden_states_139 = hidden_states_143 = None
        item_100 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_eps = (
            None
        )
        hidden_states_145 = torch.nn.functional.layer_norm(
            hidden_states_144,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            item_100,
        )
        hidden_states_144 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = (item_100) = (
            None
        )
        linear_113 = torch._C._nn.linear(
            hidden_states_145,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        item_101 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_scaling = (
            None
        )
        query_states_24 = linear_113 * item_101
        linear_113 = item_101 = None
        key_states_24 = torch._C._nn.linear(
            hidden_states_145,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_24 = torch._C._nn.linear(
            hidden_states_145,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_71 = key_states_24.view(1, -1, 16, 64)
        key_states_24 = None
        key_states_25 = view_71.transpose(1, 2)
        view_71 = None
        view_72 = value_states_24.view(1, -1, 16, 64)
        value_states_24 = None
        value_states_25 = view_72.transpose(1, 2)
        view_72 = None
        view_73 = query_states_24.view(1, 1, 16, 64)
        query_states_24 = None
        query_states_25 = view_73.transpose(1, 2)
        view_73 = None
        query_states_26 = query_states_25.reshape(16, -1, 64)
        query_states_25 = None
        key_states_26 = key_states_25.reshape(16, -1, 64)
        key_states_25 = None
        value_states_26 = value_states_25.reshape(16, -1, 64)
        value_states_25 = None
        transpose_92 = key_states_26.transpose(1, 2)
        key_states_26 = None
        attn_weights_16 = torch.bmm(query_states_26, transpose_92)
        query_states_26 = transpose_92 = None
        attn_weights_17 = torch.nn.functional.softmax(attn_weights_16, dim=-1)
        attn_weights_16 = None
        item_102 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_dropout = (
            None
        )
        attn_probs_8 = torch.nn.functional.dropout(
            attn_weights_17, p=item_102, training=False
        )
        attn_weights_17 = item_102 = None
        attn_output_64 = torch.bmm(attn_probs_8, value_states_26)
        attn_probs_8 = value_states_26 = None
        attn_output_65 = attn_output_64.view(1, 16, 1, 64)
        attn_output_64 = None
        attn_output_66 = attn_output_65.transpose(1, 2)
        attn_output_65 = None
        attn_output_67 = attn_output_66.reshape(1, 1, 1024)
        attn_output_66 = None
        attn_output_68 = torch._C._nn.linear(
            attn_output_67,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_67 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        item_103 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_dropout = (
            None
        )
        hidden_states_146 = torch.nn.functional.dropout(
            attn_output_68, p=item_103, training=False
        )
        attn_output_68 = None
        hidden_states_147 = hidden_states_145 + hidden_states_146
        hidden_states_145 = hidden_states_146 = None
        item_104 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_eps = (
            None
        )
        hidden_states_148 = torch.nn.functional.layer_norm(
            hidden_states_147,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            item_104,
        )
        hidden_states_147 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (item_104) = (
            None
        )
        linear_117 = torch._C._nn.linear(
            hidden_states_148,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        item_105 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_scaling = (
            None
        )
        query_states_27 = linear_117 * item_105
        linear_117 = item_105 = None
        key_states_27 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_27 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_75 = key_states_27.view(1, -1, 16, 64)
        key_states_27 = None
        key_states_28 = view_75.transpose(1, 2)
        view_75 = None
        view_76 = value_states_27.view(1, -1, 16, 64)
        value_states_27 = None
        value_states_28 = view_76.transpose(1, 2)
        view_76 = None
        view_77 = query_states_27.view(1, 1, 16, 64)
        query_states_27 = None
        query_states_28 = view_77.transpose(1, 2)
        view_77 = None
        query_states_29 = query_states_28.reshape(16, -1, 64)
        query_states_28 = None
        key_states_29 = key_states_28.reshape(16, -1, 64)
        key_states_28 = None
        value_states_29 = value_states_28.reshape(16, -1, 64)
        value_states_28 = None
        transpose_97 = key_states_29.transpose(1, 2)
        key_states_29 = None
        attn_weights_18 = torch.bmm(query_states_29, transpose_97)
        query_states_29 = transpose_97 = None
        attn_weights_19 = torch.nn.functional.softmax(attn_weights_18, dim=-1)
        attn_weights_18 = None
        item_106 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_dropout = (
            None
        )
        attn_probs_9 = torch.nn.functional.dropout(
            attn_weights_19, p=item_106, training=False
        )
        attn_weights_19 = item_106 = None
        attn_output_69 = torch.bmm(attn_probs_9, value_states_29)
        attn_probs_9 = value_states_29 = None
        attn_output_70 = attn_output_69.view(1, 16, 1, 64)
        attn_output_69 = None
        attn_output_71 = attn_output_70.transpose(1, 2)
        attn_output_70 = None
        attn_output_72 = attn_output_71.reshape(1, 1, 1024)
        attn_output_71 = None
        attn_output_73 = torch._C._nn.linear(
            attn_output_72,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_72 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_149 = torch.nn.functional.dropout(
            attn_output_73, p=item_103, training=False
        )
        attn_output_73 = None
        hidden_states_150 = hidden_states_148 + hidden_states_149
        hidden_states_148 = hidden_states_149 = None
        item_107 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_eps = (
            None
        )
        hidden_states_151 = torch.nn.functional.layer_norm(
            hidden_states_150,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_,
            item_107,
        )
        hidden_states_150 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_ = (item_107) = (
            None
        )
        linear_121 = torch._C._nn.linear(
            hidden_states_151,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_ = (None)
        hidden_states_152 = torch.nn.functional.relu(linear_121, inplace=False)
        linear_121 = None
        item_108 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_activation_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_activation_dropout = (
            None
        )
        hidden_states_153 = torch.nn.functional.dropout(
            hidden_states_152, p=item_108, training=False
        )
        hidden_states_152 = item_108 = None
        hidden_states_154 = torch._C._nn.linear(
            hidden_states_153,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_,
        )
        hidden_states_153 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_ = (None)
        hidden_states_155 = torch.nn.functional.dropout(
            hidden_states_154, p=item_103, training=False
        )
        hidden_states_154 = item_103 = None
        hidden_states_156 = hidden_states_151 + hidden_states_155
        hidden_states_151 = hidden_states_155 = None
        item_109 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_eps = (
            None
        )
        hidden_states_157 = torch.nn.functional.layer_norm(
            hidden_states_156,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            item_109,
        )
        hidden_states_156 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = (item_109) = (
            None
        )
        linear_123 = torch._C._nn.linear(
            hidden_states_157,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        item_110 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_scaling = (
            None
        )
        query_states_30 = linear_123 * item_110
        linear_123 = item_110 = None
        key_states_30 = torch._C._nn.linear(
            hidden_states_157,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_30 = torch._C._nn.linear(
            hidden_states_157,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_79 = key_states_30.view(1, -1, 16, 64)
        key_states_30 = None
        key_states_31 = view_79.transpose(1, 2)
        view_79 = None
        view_80 = value_states_30.view(1, -1, 16, 64)
        value_states_30 = None
        value_states_31 = view_80.transpose(1, 2)
        view_80 = None
        view_81 = query_states_30.view(1, 1, 16, 64)
        query_states_30 = None
        query_states_31 = view_81.transpose(1, 2)
        view_81 = None
        query_states_32 = query_states_31.reshape(16, -1, 64)
        query_states_31 = None
        key_states_32 = key_states_31.reshape(16, -1, 64)
        key_states_31 = None
        value_states_32 = value_states_31.reshape(16, -1, 64)
        value_states_31 = None
        transpose_102 = key_states_32.transpose(1, 2)
        key_states_32 = None
        attn_weights_20 = torch.bmm(query_states_32, transpose_102)
        query_states_32 = transpose_102 = None
        attn_weights_21 = torch.nn.functional.softmax(attn_weights_20, dim=-1)
        attn_weights_20 = None
        item_111 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_dropout = (
            None
        )
        attn_probs_10 = torch.nn.functional.dropout(
            attn_weights_21, p=item_111, training=False
        )
        attn_weights_21 = item_111 = None
        attn_output_74 = torch.bmm(attn_probs_10, value_states_32)
        attn_probs_10 = value_states_32 = None
        attn_output_75 = attn_output_74.view(1, 16, 1, 64)
        attn_output_74 = None
        attn_output_76 = attn_output_75.transpose(1, 2)
        attn_output_75 = None
        attn_output_77 = attn_output_76.reshape(1, 1, 1024)
        attn_output_76 = None
        attn_output_78 = torch._C._nn.linear(
            attn_output_77,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_77 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        item_112 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_dropout = (
            None
        )
        hidden_states_158 = torch.nn.functional.dropout(
            attn_output_78, p=item_112, training=False
        )
        attn_output_78 = None
        hidden_states_159 = hidden_states_157 + hidden_states_158
        hidden_states_157 = hidden_states_158 = None
        item_113 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_eps = (
            None
        )
        hidden_states_160 = torch.nn.functional.layer_norm(
            hidden_states_159,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_,
            item_113,
        )
        hidden_states_159 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = (item_113) = (
            None
        )
        linear_127 = torch._C._nn.linear(
            hidden_states_160,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        item_114 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_scaling = (
            None
        )
        query_states_33 = linear_127 * item_114
        linear_127 = item_114 = None
        key_states_33 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_33 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_83 = key_states_33.view(1, -1, 16, 64)
        key_states_33 = None
        key_states_34 = view_83.transpose(1, 2)
        view_83 = None
        view_84 = value_states_33.view(1, -1, 16, 64)
        value_states_33 = None
        value_states_34 = view_84.transpose(1, 2)
        view_84 = None
        view_85 = query_states_33.view(1, 1, 16, 64)
        query_states_33 = None
        query_states_34 = view_85.transpose(1, 2)
        view_85 = None
        query_states_35 = query_states_34.reshape(16, -1, 64)
        query_states_34 = None
        key_states_35 = key_states_34.reshape(16, -1, 64)
        key_states_34 = None
        value_states_35 = value_states_34.reshape(16, -1, 64)
        value_states_34 = None
        transpose_107 = key_states_35.transpose(1, 2)
        key_states_35 = None
        attn_weights_22 = torch.bmm(query_states_35, transpose_107)
        query_states_35 = transpose_107 = None
        attn_weights_23 = torch.nn.functional.softmax(attn_weights_22, dim=-1)
        attn_weights_22 = None
        item_115 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_dropout = (
            None
        )
        attn_probs_11 = torch.nn.functional.dropout(
            attn_weights_23, p=item_115, training=False
        )
        attn_weights_23 = item_115 = None
        attn_output_79 = torch.bmm(attn_probs_11, value_states_35)
        attn_probs_11 = value_states_35 = None
        attn_output_80 = attn_output_79.view(1, 16, 1, 64)
        attn_output_79 = None
        attn_output_81 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_82 = attn_output_81.reshape(1, 1, 1024)
        attn_output_81 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_82 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_161 = torch.nn.functional.dropout(
            attn_output_83, p=item_112, training=False
        )
        attn_output_83 = None
        hidden_states_162 = hidden_states_160 + hidden_states_161
        hidden_states_160 = hidden_states_161 = None
        item_116 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_eps = (
            None
        )
        hidden_states_163 = torch.nn.functional.layer_norm(
            hidden_states_162,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_,
            item_116,
        )
        hidden_states_162 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_ = (item_116) = (
            None
        )
        linear_131 = torch._C._nn.linear(
            hidden_states_163,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_ = (None)
        hidden_states_164 = torch.nn.functional.relu(linear_131, inplace=False)
        linear_131 = None
        item_117 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_activation_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_activation_dropout = (
            None
        )
        hidden_states_165 = torch.nn.functional.dropout(
            hidden_states_164, p=item_117, training=False
        )
        hidden_states_164 = item_117 = None
        hidden_states_166 = torch._C._nn.linear(
            hidden_states_165,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_,
        )
        hidden_states_165 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_ = (None)
        hidden_states_167 = torch.nn.functional.dropout(
            hidden_states_166, p=item_112, training=False
        )
        hidden_states_166 = item_112 = None
        hidden_states_168 = hidden_states_163 + hidden_states_167
        hidden_states_163 = hidden_states_167 = None
        item_118 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_eps = (
            None
        )
        hidden_states_169 = torch.nn.functional.layer_norm(
            hidden_states_168,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_,
            item_118,
        )
        hidden_states_168 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = (item_118) = (
            None
        )
        linear_133 = torch._C._nn.linear(
            hidden_states_169,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        item_119 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_scaling = (
            None
        )
        query_states_36 = linear_133 * item_119
        linear_133 = item_119 = None
        key_states_36 = torch._C._nn.linear(
            hidden_states_169,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_36 = torch._C._nn.linear(
            hidden_states_169,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_87 = key_states_36.view(1, -1, 16, 64)
        key_states_36 = None
        key_states_37 = view_87.transpose(1, 2)
        view_87 = None
        view_88 = value_states_36.view(1, -1, 16, 64)
        value_states_36 = None
        value_states_37 = view_88.transpose(1, 2)
        view_88 = None
        view_89 = query_states_36.view(1, 1, 16, 64)
        query_states_36 = None
        query_states_37 = view_89.transpose(1, 2)
        view_89 = None
        query_states_38 = query_states_37.reshape(16, -1, 64)
        query_states_37 = None
        key_states_38 = key_states_37.reshape(16, -1, 64)
        key_states_37 = None
        value_states_38 = value_states_37.reshape(16, -1, 64)
        value_states_37 = None
        transpose_112 = key_states_38.transpose(1, 2)
        key_states_38 = None
        attn_weights_24 = torch.bmm(query_states_38, transpose_112)
        query_states_38 = transpose_112 = None
        attn_weights_25 = torch.nn.functional.softmax(attn_weights_24, dim=-1)
        attn_weights_24 = None
        item_120 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_dropout = (
            None
        )
        attn_probs_12 = torch.nn.functional.dropout(
            attn_weights_25, p=item_120, training=False
        )
        attn_weights_25 = item_120 = None
        attn_output_84 = torch.bmm(attn_probs_12, value_states_38)
        attn_probs_12 = value_states_38 = None
        attn_output_85 = attn_output_84.view(1, 16, 1, 64)
        attn_output_84 = None
        attn_output_86 = attn_output_85.transpose(1, 2)
        attn_output_85 = None
        attn_output_87 = attn_output_86.reshape(1, 1, 1024)
        attn_output_86 = None
        attn_output_88 = torch._C._nn.linear(
            attn_output_87,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_87 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        item_121 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_dropout = (
            None
        )
        hidden_states_170 = torch.nn.functional.dropout(
            attn_output_88, p=item_121, training=False
        )
        attn_output_88 = None
        hidden_states_171 = hidden_states_169 + hidden_states_170
        hidden_states_169 = hidden_states_170 = None
        item_122 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_eps = (
            None
        )
        hidden_states_172 = torch.nn.functional.layer_norm(
            hidden_states_171,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_,
            item_122,
        )
        hidden_states_171 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = (item_122) = (
            None
        )
        linear_137 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        item_123 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_scaling = (
            None
        )
        query_states_39 = linear_137 * item_123
        linear_137 = item_123 = None
        key_states_39 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_39 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_91 = key_states_39.view(1, -1, 16, 64)
        key_states_39 = None
        key_states_40 = view_91.transpose(1, 2)
        view_91 = None
        view_92 = value_states_39.view(1, -1, 16, 64)
        value_states_39 = None
        value_states_40 = view_92.transpose(1, 2)
        view_92 = None
        view_93 = query_states_39.view(1, 1, 16, 64)
        query_states_39 = None
        query_states_40 = view_93.transpose(1, 2)
        view_93 = None
        query_states_41 = query_states_40.reshape(16, -1, 64)
        query_states_40 = None
        key_states_41 = key_states_40.reshape(16, -1, 64)
        key_states_40 = None
        value_states_41 = value_states_40.reshape(16, -1, 64)
        value_states_40 = None
        transpose_117 = key_states_41.transpose(1, 2)
        key_states_41 = None
        attn_weights_26 = torch.bmm(query_states_41, transpose_117)
        query_states_41 = transpose_117 = None
        attn_weights_27 = torch.nn.functional.softmax(attn_weights_26, dim=-1)
        attn_weights_26 = None
        item_124 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_dropout = (
            None
        )
        attn_probs_13 = torch.nn.functional.dropout(
            attn_weights_27, p=item_124, training=False
        )
        attn_weights_27 = item_124 = None
        attn_output_89 = torch.bmm(attn_probs_13, value_states_41)
        attn_probs_13 = value_states_41 = None
        attn_output_90 = attn_output_89.view(1, 16, 1, 64)
        attn_output_89 = None
        attn_output_91 = attn_output_90.transpose(1, 2)
        attn_output_90 = None
        attn_output_92 = attn_output_91.reshape(1, 1, 1024)
        attn_output_91 = None
        attn_output_93 = torch._C._nn.linear(
            attn_output_92,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_92 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_173 = torch.nn.functional.dropout(
            attn_output_93, p=item_121, training=False
        )
        attn_output_93 = None
        hidden_states_174 = hidden_states_172 + hidden_states_173
        hidden_states_172 = hidden_states_173 = None
        item_125 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_eps = (
            None
        )
        hidden_states_175 = torch.nn.functional.layer_norm(
            hidden_states_174,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_,
            item_125,
        )
        hidden_states_174 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_encoder_attn_layer_norm_parameters_bias_ = (item_125) = (
            None
        )
        linear_141 = torch._C._nn.linear(
            hidden_states_175,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc1_parameters_bias_ = (None)
        hidden_states_176 = torch.nn.functional.relu(linear_141, inplace=False)
        linear_141 = None
        item_126 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_activation_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_activation_dropout = (
            None
        )
        hidden_states_177 = torch.nn.functional.dropout(
            hidden_states_176, p=item_126, training=False
        )
        hidden_states_176 = item_126 = None
        hidden_states_178 = torch._C._nn.linear(
            hidden_states_177,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_,
        )
        hidden_states_177 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_fc2_parameters_bias_ = (None)
        hidden_states_179 = torch.nn.functional.dropout(
            hidden_states_178, p=item_121, training=False
        )
        hidden_states_178 = item_121 = None
        hidden_states_180 = hidden_states_175 + hidden_states_179
        hidden_states_175 = hidden_states_179 = None
        item_127 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_eps = (
            None
        )
        hidden_states_181 = torch.nn.functional.layer_norm(
            hidden_states_180,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_,
            item_127,
        )
        hidden_states_180 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_ = (item_127) = (
            None
        )
        linear_143 = torch._C._nn.linear(
            hidden_states_181,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        item_128 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_scaling = (
            None
        )
        query_states_42 = linear_143 * item_128
        linear_143 = item_128 = None
        key_states_42 = torch._C._nn.linear(
            hidden_states_181,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_42 = torch._C._nn.linear(
            hidden_states_181,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_95 = key_states_42.view(1, -1, 16, 64)
        key_states_42 = None
        key_states_43 = view_95.transpose(1, 2)
        view_95 = None
        view_96 = value_states_42.view(1, -1, 16, 64)
        value_states_42 = None
        value_states_43 = view_96.transpose(1, 2)
        view_96 = None
        view_97 = query_states_42.view(1, 1, 16, 64)
        query_states_42 = None
        query_states_43 = view_97.transpose(1, 2)
        view_97 = None
        query_states_44 = query_states_43.reshape(16, -1, 64)
        query_states_43 = None
        key_states_44 = key_states_43.reshape(16, -1, 64)
        key_states_43 = None
        value_states_44 = value_states_43.reshape(16, -1, 64)
        value_states_43 = None
        transpose_122 = key_states_44.transpose(1, 2)
        key_states_44 = None
        attn_weights_28 = torch.bmm(query_states_44, transpose_122)
        query_states_44 = transpose_122 = None
        attn_weights_29 = torch.nn.functional.softmax(attn_weights_28, dim=-1)
        attn_weights_28 = None
        item_129 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_dropout = (
            None
        )
        attn_probs_14 = torch.nn.functional.dropout(
            attn_weights_29, p=item_129, training=False
        )
        attn_weights_29 = item_129 = None
        attn_output_94 = torch.bmm(attn_probs_14, value_states_44)
        attn_probs_14 = value_states_44 = None
        attn_output_95 = attn_output_94.view(1, 16, 1, 64)
        attn_output_94 = None
        attn_output_96 = attn_output_95.transpose(1, 2)
        attn_output_95 = None
        attn_output_97 = attn_output_96.reshape(1, 1, 1024)
        attn_output_96 = None
        attn_output_98 = torch._C._nn.linear(
            attn_output_97,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_97 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        item_130 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_dropout = (
            None
        )
        hidden_states_182 = torch.nn.functional.dropout(
            attn_output_98, p=item_130, training=False
        )
        attn_output_98 = None
        hidden_states_183 = hidden_states_181 + hidden_states_182
        hidden_states_181 = hidden_states_182 = None
        item_131 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_eps = (
            None
        )
        hidden_states_184 = torch.nn.functional.layer_norm(
            hidden_states_183,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_,
            item_131,
        )
        hidden_states_183 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_ = (item_131) = (
            None
        )
        linear_147 = torch._C._nn.linear(
            hidden_states_184,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        item_132 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_scaling = (
            None
        )
        query_states_45 = linear_147 * item_132
        linear_147 = item_132 = None
        key_states_45 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_45 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_99 = key_states_45.view(1, -1, 16, 64)
        key_states_45 = None
        key_states_46 = view_99.transpose(1, 2)
        view_99 = None
        view_100 = value_states_45.view(1, -1, 16, 64)
        value_states_45 = None
        value_states_46 = view_100.transpose(1, 2)
        view_100 = None
        view_101 = query_states_45.view(1, 1, 16, 64)
        query_states_45 = None
        query_states_46 = view_101.transpose(1, 2)
        view_101 = None
        query_states_47 = query_states_46.reshape(16, -1, 64)
        query_states_46 = None
        key_states_47 = key_states_46.reshape(16, -1, 64)
        key_states_46 = None
        value_states_47 = value_states_46.reshape(16, -1, 64)
        value_states_46 = None
        transpose_127 = key_states_47.transpose(1, 2)
        key_states_47 = None
        attn_weights_30 = torch.bmm(query_states_47, transpose_127)
        query_states_47 = transpose_127 = None
        attn_weights_31 = torch.nn.functional.softmax(attn_weights_30, dim=-1)
        attn_weights_30 = None
        item_133 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_dropout = (
            None
        )
        attn_probs_15 = torch.nn.functional.dropout(
            attn_weights_31, p=item_133, training=False
        )
        attn_weights_31 = item_133 = None
        attn_output_99 = torch.bmm(attn_probs_15, value_states_47)
        attn_probs_15 = value_states_47 = None
        attn_output_100 = attn_output_99.view(1, 16, 1, 64)
        attn_output_99 = None
        attn_output_101 = attn_output_100.transpose(1, 2)
        attn_output_100 = None
        attn_output_102 = attn_output_101.reshape(1, 1, 1024)
        attn_output_101 = None
        attn_output_103 = torch._C._nn.linear(
            attn_output_102,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_102 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_185 = torch.nn.functional.dropout(
            attn_output_103, p=item_130, training=False
        )
        attn_output_103 = None
        hidden_states_186 = hidden_states_184 + hidden_states_185
        hidden_states_184 = hidden_states_185 = None
        item_134 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_eps = (
            None
        )
        hidden_states_187 = torch.nn.functional.layer_norm(
            hidden_states_186,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_,
            item_134,
        )
        hidden_states_186 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_encoder_attn_layer_norm_parameters_bias_ = (item_134) = (
            None
        )
        linear_151 = torch._C._nn.linear(
            hidden_states_187,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc1_parameters_bias_ = (None)
        hidden_states_188 = torch.nn.functional.relu(linear_151, inplace=False)
        linear_151 = None
        item_135 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_activation_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_activation_dropout = (
            None
        )
        hidden_states_189 = torch.nn.functional.dropout(
            hidden_states_188, p=item_135, training=False
        )
        hidden_states_188 = item_135 = None
        hidden_states_190 = torch._C._nn.linear(
            hidden_states_189,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_,
        )
        hidden_states_189 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_fc2_parameters_bias_ = (None)
        hidden_states_191 = torch.nn.functional.dropout(
            hidden_states_190, p=item_130, training=False
        )
        hidden_states_190 = item_130 = None
        hidden_states_192 = hidden_states_187 + hidden_states_191
        hidden_states_187 = hidden_states_191 = None
        item_136 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_eps = (
            None
        )
        hidden_states_193 = torch.nn.functional.layer_norm(
            hidden_states_192,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_,
            item_136,
        )
        hidden_states_192 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_ = (item_136) = (
            None
        )
        linear_153 = torch._C._nn.linear(
            hidden_states_193,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        item_137 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_scaling = (
            None
        )
        query_states_48 = linear_153 * item_137
        linear_153 = item_137 = None
        key_states_48 = torch._C._nn.linear(
            hidden_states_193,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_48 = torch._C._nn.linear(
            hidden_states_193,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_103 = key_states_48.view(1, -1, 16, 64)
        key_states_48 = None
        key_states_49 = view_103.transpose(1, 2)
        view_103 = None
        view_104 = value_states_48.view(1, -1, 16, 64)
        value_states_48 = None
        value_states_49 = view_104.transpose(1, 2)
        view_104 = None
        view_105 = query_states_48.view(1, 1, 16, 64)
        query_states_48 = None
        query_states_49 = view_105.transpose(1, 2)
        view_105 = None
        query_states_50 = query_states_49.reshape(16, -1, 64)
        query_states_49 = None
        key_states_50 = key_states_49.reshape(16, -1, 64)
        key_states_49 = None
        value_states_50 = value_states_49.reshape(16, -1, 64)
        value_states_49 = None
        transpose_132 = key_states_50.transpose(1, 2)
        key_states_50 = None
        attn_weights_32 = torch.bmm(query_states_50, transpose_132)
        query_states_50 = transpose_132 = None
        attn_weights_33 = torch.nn.functional.softmax(attn_weights_32, dim=-1)
        attn_weights_32 = None
        item_138 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_dropout = (
            None
        )
        attn_probs_16 = torch.nn.functional.dropout(
            attn_weights_33, p=item_138, training=False
        )
        attn_weights_33 = item_138 = None
        attn_output_104 = torch.bmm(attn_probs_16, value_states_50)
        attn_probs_16 = value_states_50 = None
        attn_output_105 = attn_output_104.view(1, 16, 1, 64)
        attn_output_104 = None
        attn_output_106 = attn_output_105.transpose(1, 2)
        attn_output_105 = None
        attn_output_107 = attn_output_106.reshape(1, 1, 1024)
        attn_output_106 = None
        attn_output_108 = torch._C._nn.linear(
            attn_output_107,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_107 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        item_139 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_dropout = (
            None
        )
        hidden_states_194 = torch.nn.functional.dropout(
            attn_output_108, p=item_139, training=False
        )
        attn_output_108 = None
        hidden_states_195 = hidden_states_193 + hidden_states_194
        hidden_states_193 = hidden_states_194 = None
        item_140 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_eps = (
            None
        )
        hidden_states_196 = torch.nn.functional.layer_norm(
            hidden_states_195,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_,
            item_140,
        )
        hidden_states_195 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_ = (item_140) = (
            None
        )
        linear_157 = torch._C._nn.linear(
            hidden_states_196,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        item_141 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_scaling = (
            None
        )
        query_states_51 = linear_157 * item_141
        linear_157 = item_141 = None
        key_states_51 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_51 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_107 = key_states_51.view(1, -1, 16, 64)
        key_states_51 = None
        key_states_52 = view_107.transpose(1, 2)
        view_107 = None
        view_108 = value_states_51.view(1, -1, 16, 64)
        value_states_51 = None
        value_states_52 = view_108.transpose(1, 2)
        view_108 = None
        view_109 = query_states_51.view(1, 1, 16, 64)
        query_states_51 = None
        query_states_52 = view_109.transpose(1, 2)
        view_109 = None
        query_states_53 = query_states_52.reshape(16, -1, 64)
        query_states_52 = None
        key_states_53 = key_states_52.reshape(16, -1, 64)
        key_states_52 = None
        value_states_53 = value_states_52.reshape(16, -1, 64)
        value_states_52 = None
        transpose_137 = key_states_53.transpose(1, 2)
        key_states_53 = None
        attn_weights_34 = torch.bmm(query_states_53, transpose_137)
        query_states_53 = transpose_137 = None
        attn_weights_35 = torch.nn.functional.softmax(attn_weights_34, dim=-1)
        attn_weights_34 = None
        item_142 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_dropout = (
            None
        )
        attn_probs_17 = torch.nn.functional.dropout(
            attn_weights_35, p=item_142, training=False
        )
        attn_weights_35 = item_142 = None
        attn_output_109 = torch.bmm(attn_probs_17, value_states_53)
        attn_probs_17 = value_states_53 = None
        attn_output_110 = attn_output_109.view(1, 16, 1, 64)
        attn_output_109 = None
        attn_output_111 = attn_output_110.transpose(1, 2)
        attn_output_110 = None
        attn_output_112 = attn_output_111.reshape(1, 1, 1024)
        attn_output_111 = None
        attn_output_113 = torch._C._nn.linear(
            attn_output_112,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_112 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_197 = torch.nn.functional.dropout(
            attn_output_113, p=item_139, training=False
        )
        attn_output_113 = None
        hidden_states_198 = hidden_states_196 + hidden_states_197
        hidden_states_196 = hidden_states_197 = None
        item_143 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_eps = (
            None
        )
        hidden_states_199 = torch.nn.functional.layer_norm(
            hidden_states_198,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_,
            item_143,
        )
        hidden_states_198 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_encoder_attn_layer_norm_parameters_bias_ = (item_143) = (
            None
        )
        linear_161 = torch._C._nn.linear(
            hidden_states_199,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc1_parameters_bias_ = (None)
        hidden_states_200 = torch.nn.functional.relu(linear_161, inplace=False)
        linear_161 = None
        item_144 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_activation_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_activation_dropout = (
            None
        )
        hidden_states_201 = torch.nn.functional.dropout(
            hidden_states_200, p=item_144, training=False
        )
        hidden_states_200 = item_144 = None
        hidden_states_202 = torch._C._nn.linear(
            hidden_states_201,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_,
        )
        hidden_states_201 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_fc2_parameters_bias_ = (None)
        hidden_states_203 = torch.nn.functional.dropout(
            hidden_states_202, p=item_139, training=False
        )
        hidden_states_202 = item_139 = None
        hidden_states_204 = hidden_states_199 + hidden_states_203
        hidden_states_199 = hidden_states_203 = None
        item_145 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_eps = (
            None
        )
        hidden_states_205 = torch.nn.functional.layer_norm(
            hidden_states_204,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_,
            item_145,
        )
        hidden_states_204 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_ = (item_145) = (
            None
        )
        linear_163 = torch._C._nn.linear(
            hidden_states_205,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        item_146 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_scaling = (
            None
        )
        query_states_54 = linear_163 * item_146
        linear_163 = item_146 = None
        key_states_54 = torch._C._nn.linear(
            hidden_states_205,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_54 = torch._C._nn.linear(
            hidden_states_205,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_111 = key_states_54.view(1, -1, 16, 64)
        key_states_54 = None
        key_states_55 = view_111.transpose(1, 2)
        view_111 = None
        view_112 = value_states_54.view(1, -1, 16, 64)
        value_states_54 = None
        value_states_55 = view_112.transpose(1, 2)
        view_112 = None
        view_113 = query_states_54.view(1, 1, 16, 64)
        query_states_54 = None
        query_states_55 = view_113.transpose(1, 2)
        view_113 = None
        query_states_56 = query_states_55.reshape(16, -1, 64)
        query_states_55 = None
        key_states_56 = key_states_55.reshape(16, -1, 64)
        key_states_55 = None
        value_states_56 = value_states_55.reshape(16, -1, 64)
        value_states_55 = None
        transpose_142 = key_states_56.transpose(1, 2)
        key_states_56 = None
        attn_weights_36 = torch.bmm(query_states_56, transpose_142)
        query_states_56 = transpose_142 = None
        attn_weights_37 = torch.nn.functional.softmax(attn_weights_36, dim=-1)
        attn_weights_36 = None
        item_147 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_dropout = (
            None
        )
        attn_probs_18 = torch.nn.functional.dropout(
            attn_weights_37, p=item_147, training=False
        )
        attn_weights_37 = item_147 = None
        attn_output_114 = torch.bmm(attn_probs_18, value_states_56)
        attn_probs_18 = value_states_56 = None
        attn_output_115 = attn_output_114.view(1, 16, 1, 64)
        attn_output_114 = None
        attn_output_116 = attn_output_115.transpose(1, 2)
        attn_output_115 = None
        attn_output_117 = attn_output_116.reshape(1, 1, 1024)
        attn_output_116 = None
        attn_output_118 = torch._C._nn.linear(
            attn_output_117,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_117 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        item_148 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_dropout = (
            None
        )
        hidden_states_206 = torch.nn.functional.dropout(
            attn_output_118, p=item_148, training=False
        )
        attn_output_118 = None
        hidden_states_207 = hidden_states_205 + hidden_states_206
        hidden_states_205 = hidden_states_206 = None
        item_149 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_eps = (
            None
        )
        hidden_states_208 = torch.nn.functional.layer_norm(
            hidden_states_207,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_,
            item_149,
        )
        hidden_states_207 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_ = (item_149) = (
            None
        )
        linear_167 = torch._C._nn.linear(
            hidden_states_208,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        item_150 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_scaling = (
            None
        )
        query_states_57 = linear_167 * item_150
        linear_167 = item_150 = None
        key_states_57 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_57 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_115 = key_states_57.view(1, -1, 16, 64)
        key_states_57 = None
        key_states_58 = view_115.transpose(1, 2)
        view_115 = None
        view_116 = value_states_57.view(1, -1, 16, 64)
        value_states_57 = None
        value_states_58 = view_116.transpose(1, 2)
        view_116 = None
        view_117 = query_states_57.view(1, 1, 16, 64)
        query_states_57 = None
        query_states_58 = view_117.transpose(1, 2)
        view_117 = None
        query_states_59 = query_states_58.reshape(16, -1, 64)
        query_states_58 = None
        key_states_59 = key_states_58.reshape(16, -1, 64)
        key_states_58 = None
        value_states_59 = value_states_58.reshape(16, -1, 64)
        value_states_58 = None
        transpose_147 = key_states_59.transpose(1, 2)
        key_states_59 = None
        attn_weights_38 = torch.bmm(query_states_59, transpose_147)
        query_states_59 = transpose_147 = None
        attn_weights_39 = torch.nn.functional.softmax(attn_weights_38, dim=-1)
        attn_weights_38 = None
        item_151 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_dropout = (
            None
        )
        attn_probs_19 = torch.nn.functional.dropout(
            attn_weights_39, p=item_151, training=False
        )
        attn_weights_39 = item_151 = None
        attn_output_119 = torch.bmm(attn_probs_19, value_states_59)
        attn_probs_19 = value_states_59 = None
        attn_output_120 = attn_output_119.view(1, 16, 1, 64)
        attn_output_119 = None
        attn_output_121 = attn_output_120.transpose(1, 2)
        attn_output_120 = None
        attn_output_122 = attn_output_121.reshape(1, 1, 1024)
        attn_output_121 = None
        attn_output_123 = torch._C._nn.linear(
            attn_output_122,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_122 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_209 = torch.nn.functional.dropout(
            attn_output_123, p=item_148, training=False
        )
        attn_output_123 = None
        hidden_states_210 = hidden_states_208 + hidden_states_209
        hidden_states_208 = hidden_states_209 = None
        item_152 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_eps = (
            None
        )
        hidden_states_211 = torch.nn.functional.layer_norm(
            hidden_states_210,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_,
            item_152,
        )
        hidden_states_210 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_encoder_attn_layer_norm_parameters_bias_ = (item_152) = (
            None
        )
        linear_171 = torch._C._nn.linear(
            hidden_states_211,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc1_parameters_bias_ = (None)
        hidden_states_212 = torch.nn.functional.relu(linear_171, inplace=False)
        linear_171 = None
        item_153 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_activation_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_activation_dropout = (
            None
        )
        hidden_states_213 = torch.nn.functional.dropout(
            hidden_states_212, p=item_153, training=False
        )
        hidden_states_212 = item_153 = None
        hidden_states_214 = torch._C._nn.linear(
            hidden_states_213,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_,
        )
        hidden_states_213 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_fc2_parameters_bias_ = (None)
        hidden_states_215 = torch.nn.functional.dropout(
            hidden_states_214, p=item_148, training=False
        )
        hidden_states_214 = item_148 = None
        hidden_states_216 = hidden_states_211 + hidden_states_215
        hidden_states_211 = hidden_states_215 = None
        item_154 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_eps = (
            None
        )
        hidden_states_217 = torch.nn.functional.layer_norm(
            hidden_states_216,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_,
            item_154,
        )
        hidden_states_216 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_ = (item_154) = (
            None
        )
        linear_173 = torch._C._nn.linear(
            hidden_states_217,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        item_155 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_scaling = (
            None
        )
        query_states_60 = linear_173 * item_155
        linear_173 = item_155 = None
        key_states_60 = torch._C._nn.linear(
            hidden_states_217,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_60 = torch._C._nn.linear(
            hidden_states_217,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_119 = key_states_60.view(1, -1, 16, 64)
        key_states_60 = None
        key_states_61 = view_119.transpose(1, 2)
        view_119 = None
        view_120 = value_states_60.view(1, -1, 16, 64)
        value_states_60 = None
        value_states_61 = view_120.transpose(1, 2)
        view_120 = None
        view_121 = query_states_60.view(1, 1, 16, 64)
        query_states_60 = None
        query_states_61 = view_121.transpose(1, 2)
        view_121 = None
        query_states_62 = query_states_61.reshape(16, -1, 64)
        query_states_61 = None
        key_states_62 = key_states_61.reshape(16, -1, 64)
        key_states_61 = None
        value_states_62 = value_states_61.reshape(16, -1, 64)
        value_states_61 = None
        transpose_152 = key_states_62.transpose(1, 2)
        key_states_62 = None
        attn_weights_40 = torch.bmm(query_states_62, transpose_152)
        query_states_62 = transpose_152 = None
        attn_weights_41 = torch.nn.functional.softmax(attn_weights_40, dim=-1)
        attn_weights_40 = None
        item_156 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_dropout = (
            None
        )
        attn_probs_20 = torch.nn.functional.dropout(
            attn_weights_41, p=item_156, training=False
        )
        attn_weights_41 = item_156 = None
        attn_output_124 = torch.bmm(attn_probs_20, value_states_62)
        attn_probs_20 = value_states_62 = None
        attn_output_125 = attn_output_124.view(1, 16, 1, 64)
        attn_output_124 = None
        attn_output_126 = attn_output_125.transpose(1, 2)
        attn_output_125 = None
        attn_output_127 = attn_output_126.reshape(1, 1, 1024)
        attn_output_126 = None
        attn_output_128 = torch._C._nn.linear(
            attn_output_127,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_127 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        item_157 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_dropout = (
            None
        )
        hidden_states_218 = torch.nn.functional.dropout(
            attn_output_128, p=item_157, training=False
        )
        attn_output_128 = None
        hidden_states_219 = hidden_states_217 + hidden_states_218
        hidden_states_217 = hidden_states_218 = None
        item_158 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_eps = (
            None
        )
        hidden_states_220 = torch.nn.functional.layer_norm(
            hidden_states_219,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_,
            item_158,
        )
        hidden_states_219 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_ = (item_158) = (
            None
        )
        linear_177 = torch._C._nn.linear(
            hidden_states_220,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        item_159 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_scaling = (
            None
        )
        query_states_63 = linear_177 * item_159
        linear_177 = item_159 = None
        key_states_63 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_63 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_123 = key_states_63.view(1, -1, 16, 64)
        key_states_63 = None
        key_states_64 = view_123.transpose(1, 2)
        view_123 = None
        view_124 = value_states_63.view(1, -1, 16, 64)
        value_states_63 = None
        value_states_64 = view_124.transpose(1, 2)
        view_124 = None
        view_125 = query_states_63.view(1, 1, 16, 64)
        query_states_63 = None
        query_states_64 = view_125.transpose(1, 2)
        view_125 = None
        query_states_65 = query_states_64.reshape(16, -1, 64)
        query_states_64 = None
        key_states_65 = key_states_64.reshape(16, -1, 64)
        key_states_64 = None
        value_states_65 = value_states_64.reshape(16, -1, 64)
        value_states_64 = None
        transpose_157 = key_states_65.transpose(1, 2)
        key_states_65 = None
        attn_weights_42 = torch.bmm(query_states_65, transpose_157)
        query_states_65 = transpose_157 = None
        attn_weights_43 = torch.nn.functional.softmax(attn_weights_42, dim=-1)
        attn_weights_42 = None
        item_160 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_dropout = (
            None
        )
        attn_probs_21 = torch.nn.functional.dropout(
            attn_weights_43, p=item_160, training=False
        )
        attn_weights_43 = item_160 = None
        attn_output_129 = torch.bmm(attn_probs_21, value_states_65)
        attn_probs_21 = value_states_65 = None
        attn_output_130 = attn_output_129.view(1, 16, 1, 64)
        attn_output_129 = None
        attn_output_131 = attn_output_130.transpose(1, 2)
        attn_output_130 = None
        attn_output_132 = attn_output_131.reshape(1, 1, 1024)
        attn_output_131 = None
        attn_output_133 = torch._C._nn.linear(
            attn_output_132,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_132 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_221 = torch.nn.functional.dropout(
            attn_output_133, p=item_157, training=False
        )
        attn_output_133 = None
        hidden_states_222 = hidden_states_220 + hidden_states_221
        hidden_states_220 = hidden_states_221 = None
        item_161 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_eps = (
            None
        )
        hidden_states_223 = torch.nn.functional.layer_norm(
            hidden_states_222,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_,
            item_161,
        )
        hidden_states_222 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_encoder_attn_layer_norm_parameters_bias_ = (item_161) = (
            None
        )
        linear_181 = torch._C._nn.linear(
            hidden_states_223,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc1_parameters_bias_ = (None)
        hidden_states_224 = torch.nn.functional.relu(linear_181, inplace=False)
        linear_181 = None
        item_162 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_activation_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_activation_dropout = (
            None
        )
        hidden_states_225 = torch.nn.functional.dropout(
            hidden_states_224, p=item_162, training=False
        )
        hidden_states_224 = item_162 = None
        hidden_states_226 = torch._C._nn.linear(
            hidden_states_225,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_,
        )
        hidden_states_225 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_fc2_parameters_bias_ = (None)
        hidden_states_227 = torch.nn.functional.dropout(
            hidden_states_226, p=item_157, training=False
        )
        hidden_states_226 = item_157 = None
        hidden_states_228 = hidden_states_223 + hidden_states_227
        hidden_states_223 = hidden_states_227 = None
        item_163 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_eps = (
            None
        )
        hidden_states_229 = torch.nn.functional.layer_norm(
            hidden_states_228,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_,
            item_163,
        )
        hidden_states_228 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_ = (item_163) = (
            None
        )
        linear_183 = torch._C._nn.linear(
            hidden_states_229,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        item_164 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_scaling = (
            None
        )
        query_states_66 = linear_183 * item_164
        linear_183 = item_164 = None
        key_states_66 = torch._C._nn.linear(
            hidden_states_229,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_66 = torch._C._nn.linear(
            hidden_states_229,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_127 = key_states_66.view(1, -1, 16, 64)
        key_states_66 = None
        key_states_67 = view_127.transpose(1, 2)
        view_127 = None
        view_128 = value_states_66.view(1, -1, 16, 64)
        value_states_66 = None
        value_states_67 = view_128.transpose(1, 2)
        view_128 = None
        view_129 = query_states_66.view(1, 1, 16, 64)
        query_states_66 = None
        query_states_67 = view_129.transpose(1, 2)
        view_129 = None
        query_states_68 = query_states_67.reshape(16, -1, 64)
        query_states_67 = None
        key_states_68 = key_states_67.reshape(16, -1, 64)
        key_states_67 = None
        value_states_68 = value_states_67.reshape(16, -1, 64)
        value_states_67 = None
        transpose_162 = key_states_68.transpose(1, 2)
        key_states_68 = None
        attn_weights_44 = torch.bmm(query_states_68, transpose_162)
        query_states_68 = transpose_162 = None
        attn_weights_45 = torch.nn.functional.softmax(attn_weights_44, dim=-1)
        attn_weights_44 = None
        item_165 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_dropout = (
            None
        )
        attn_probs_22 = torch.nn.functional.dropout(
            attn_weights_45, p=item_165, training=False
        )
        attn_weights_45 = item_165 = None
        attn_output_134 = torch.bmm(attn_probs_22, value_states_68)
        attn_probs_22 = value_states_68 = None
        attn_output_135 = attn_output_134.view(1, 16, 1, 64)
        attn_output_134 = None
        attn_output_136 = attn_output_135.transpose(1, 2)
        attn_output_135 = None
        attn_output_137 = attn_output_136.reshape(1, 1, 1024)
        attn_output_136 = None
        attn_output_138 = torch._C._nn.linear(
            attn_output_137,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_137 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        item_166 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_dropout = (
            None
        )
        hidden_states_230 = torch.nn.functional.dropout(
            attn_output_138, p=item_166, training=False
        )
        attn_output_138 = None
        hidden_states_231 = hidden_states_229 + hidden_states_230
        hidden_states_229 = hidden_states_230 = None
        item_167 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_eps = (
            None
        )
        hidden_states_232 = torch.nn.functional.layer_norm(
            hidden_states_231,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_,
            item_167,
        )
        hidden_states_231 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_ = (item_167) = (
            None
        )
        linear_187 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        item_168 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_scaling.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_scaling = (
            None
        )
        query_states_69 = linear_187 * item_168
        linear_187 = item_168 = None
        key_states_69 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        value_states_69 = torch._C._nn.linear(
            sequence_output,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_131 = key_states_69.view(1, -1, 16, 64)
        key_states_69 = None
        key_states_70 = view_131.transpose(1, 2)
        view_131 = None
        view_132 = value_states_69.view(1, -1, 16, 64)
        value_states_69 = None
        value_states_70 = view_132.transpose(1, 2)
        view_132 = None
        view_133 = query_states_69.view(1, 1, 16, 64)
        query_states_69 = None
        query_states_70 = view_133.transpose(1, 2)
        view_133 = None
        query_states_71 = query_states_70.reshape(16, -1, 64)
        query_states_70 = None
        key_states_71 = key_states_70.reshape(16, -1, 64)
        key_states_70 = None
        value_states_71 = value_states_70.reshape(16, -1, 64)
        value_states_70 = None
        transpose_167 = key_states_71.transpose(1, 2)
        key_states_71 = None
        attn_weights_46 = torch.bmm(query_states_71, transpose_167)
        query_states_71 = transpose_167 = None
        attn_weights_47 = torch.nn.functional.softmax(attn_weights_46, dim=-1)
        attn_weights_46 = None
        item_169 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_dropout = (
            None
        )
        attn_probs_23 = torch.nn.functional.dropout(
            attn_weights_47, p=item_169, training=False
        )
        attn_weights_47 = item_169 = None
        attn_output_139 = torch.bmm(attn_probs_23, value_states_71)
        attn_probs_23 = value_states_71 = None
        attn_output_140 = attn_output_139.view(1, 16, 1, 64)
        attn_output_139 = None
        attn_output_141 = attn_output_140.transpose(1, 2)
        attn_output_140 = None
        attn_output_142 = attn_output_141.reshape(1, 1, 1024)
        attn_output_141 = None
        attn_output_143 = torch._C._nn.linear(
            attn_output_142,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_142 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_233 = torch.nn.functional.dropout(
            attn_output_143, p=item_166, training=False
        )
        attn_output_143 = None
        hidden_states_234 = hidden_states_232 + hidden_states_233
        hidden_states_232 = hidden_states_233 = None
        item_170 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_eps = (
            None
        )
        hidden_states_235 = torch.nn.functional.layer_norm(
            hidden_states_234,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_,
            item_170,
        )
        hidden_states_234 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_encoder_attn_layer_norm_parameters_bias_ = (item_170) = (
            None
        )
        linear_191 = torch._C._nn.linear(
            hidden_states_235,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc1_parameters_bias_ = (None)
        hidden_states_236 = torch.nn.functional.relu(linear_191, inplace=False)
        linear_191 = None
        item_171 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_activation_dropout.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_activation_dropout = (
            None
        )
        hidden_states_237 = torch.nn.functional.dropout(
            hidden_states_236, p=item_171, training=False
        )
        hidden_states_236 = item_171 = None
        hidden_states_238 = torch._C._nn.linear(
            hidden_states_237,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_,
        )
        hidden_states_237 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_fc2_parameters_bias_ = (None)
        hidden_states_239 = torch.nn.functional.dropout(
            hidden_states_238, p=item_166, training=False
        )
        hidden_states_238 = item_166 = None
        hidden_states_240 = hidden_states_235 + hidden_states_239
        hidden_states_235 = hidden_states_239 = None
        item_172 = (
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_eps.item()
        )
        l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_eps = (
            None
        )
        hidden_states_241 = torch.nn.functional.layer_norm(
            hidden_states_240,
            (1024,),
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_,
            item_172,
        )
        hidden_states_240 = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_ = (item_172) = (
            None
        )
        logits = torch._C._nn.linear(
            hidden_states_241,
            l_self_modules_decoder_modules_output_projection_parameters_weight_,
            None,
        )
        hidden_states_241 = (
            l_self_modules_decoder_modules_output_projection_parameters_weight_
        ) = None
        return (to_1, logits, sequence_output)
