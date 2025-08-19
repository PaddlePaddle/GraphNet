import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_pixel_values_: torch.Tensor,
        L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_norm_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_norm_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_norm_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_norm_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_sr_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_sr_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_norm_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_norm_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_query_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_query_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_key_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_key_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_value_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_value_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_norm_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layer_norm_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_pixel_values_ = L_pixel_values_
        l_self_modules_encoder_modules_patch_embeddings_modules_0_modules_proj_parameters_weight_ = L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_proj_parameters_weight_
        l_self_modules_encoder_modules_patch_embeddings_modules_0_modules_proj_parameters_bias_ = L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_proj_parameters_bias_
        l_self_modules_encoder_modules_patch_embeddings_modules_0_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_patch_embeddings_modules_0_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_layer_norm_modules_0_parameters_weight_ = (
            L_self_modules_encoder_modules_layer_norm_modules_0_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_norm_modules_0_parameters_bias_ = (
            L_self_modules_encoder_modules_layer_norm_modules_0_parameters_bias_
        )
        l_self_modules_encoder_modules_patch_embeddings_modules_1_modules_proj_parameters_weight_ = L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_proj_parameters_weight_
        l_self_modules_encoder_modules_patch_embeddings_modules_1_modules_proj_parameters_bias_ = L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_proj_parameters_bias_
        l_self_modules_encoder_modules_patch_embeddings_modules_1_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_patch_embeddings_modules_1_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_layer_norm_modules_1_parameters_weight_ = (
            L_self_modules_encoder_modules_layer_norm_modules_1_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_norm_modules_1_parameters_bias_ = (
            L_self_modules_encoder_modules_layer_norm_modules_1_parameters_bias_
        )
        l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_weight_ = L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_weight_
        l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_bias_ = L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_bias_
        l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_sr_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_sr_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_sr_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_sr_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_layer_norm_modules_2_parameters_weight_ = (
            L_self_modules_encoder_modules_layer_norm_modules_2_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_norm_modules_2_parameters_bias_ = (
            L_self_modules_encoder_modules_layer_norm_modules_2_parameters_bias_
        )
        l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_weight_ = L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_weight_
        l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_bias_ = L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_bias_
        l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_query_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_query_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_key_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_key_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_value_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_value_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_2_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense1_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense1_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense1_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense1_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense2_parameters_weight_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense2_parameters_weight_
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense2_parameters_bias_ = L_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense2_parameters_bias_
        l_self_modules_encoder_modules_layer_norm_modules_3_parameters_weight_ = (
            L_self_modules_encoder_modules_layer_norm_modules_3_parameters_weight_
        )
        l_self_modules_encoder_modules_layer_norm_modules_3_parameters_bias_ = (
            L_self_modules_encoder_modules_layer_norm_modules_3_parameters_bias_
        )
        embeddings = torch.conv2d(
            l_pixel_values_,
            l_self_modules_encoder_modules_patch_embeddings_modules_0_modules_proj_parameters_weight_,
            l_self_modules_encoder_modules_patch_embeddings_modules_0_modules_proj_parameters_bias_,
            (4, 4),
            (3, 3),
            (1, 1),
            1,
        )
        l_pixel_values_ = l_self_modules_encoder_modules_patch_embeddings_modules_0_modules_proj_parameters_weight_ = l_self_modules_encoder_modules_patch_embeddings_modules_0_modules_proj_parameters_bias_ = (None)
        flatten = embeddings.flatten(2)
        embeddings = None
        embeddings_1 = flatten.transpose(1, 2)
        flatten = None
        embeddings_2 = torch.nn.functional.layer_norm(
            embeddings_1,
            (64,),
            l_self_modules_encoder_modules_patch_embeddings_modules_0_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_patch_embeddings_modules_0_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        embeddings_1 = l_self_modules_encoder_modules_patch_embeddings_modules_0_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_patch_embeddings_modules_0_modules_layer_norm_parameters_bias_ = (None)
        layer_norm_1 = torch.nn.functional.layer_norm(
            embeddings_2,
            (64,),
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_1_parameters_bias_ = (None)
        linear = torch._C._nn.linear(
            layer_norm_1,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view = linear.view(1, -1, 1, 64)
        linear = None
        query_layer = view.transpose(1, 2)
        view = None
        permute = layer_norm_1.permute(0, 2, 1)
        layer_norm_1 = None
        hidden_states = permute.reshape(1, 64, 128, 128)
        permute = None
        hidden_states_1 = torch.conv2d(
            hidden_states,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_,
            (8, 8),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_1 = hidden_states_1.reshape(1, 64, -1)
        hidden_states_1 = None
        hidden_states_2 = reshape_1.permute(0, 2, 1)
        reshape_1 = None
        hidden_states_3 = torch.nn.functional.layer_norm(
            hidden_states_2,
            (64,),
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_2 = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_1 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_1 = linear_1.view(1, -1, 1, 64)
        linear_1 = None
        key_layer = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            hidden_states_3,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_3 = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_2 = linear_2.view(1, -1, 1, 64)
        linear_2 = None
        value_layer = view_2.transpose(1, 2)
        view_2 = None
        transpose_4 = key_layer.transpose(-1, -2)
        key_layer = None
        attention_scores = torch.matmul(query_layer, transpose_4)
        query_layer = transpose_4 = None
        attention_scores_1 = attention_scores / 8.0
        attention_scores = None
        attention_probs = torch.nn.functional.softmax(attention_scores_1, dim=-1)
        attention_scores_1 = None
        attention_probs_1 = torch.nn.functional.dropout(
            attention_probs, 0.0, False, False
        )
        attention_probs = None
        context_layer = torch.matmul(attention_probs_1, value_layer)
        attention_probs_1 = value_layer = None
        permute_2 = context_layer.permute(0, 2, 1, 3)
        context_layer = None
        context_layer_1 = permute_2.contiguous()
        permute_2 = None
        context_layer_2 = context_layer_1.view((1, 16384, 64))
        context_layer_1 = None
        hidden_states_4 = torch._C._nn.linear(
            context_layer_2,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_2 = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_5 = torch.nn.functional.dropout(
            hidden_states_4, 0.0, False, False
        )
        hidden_states_4 = None
        hidden_states_6 = hidden_states_5 + embeddings_2
        hidden_states_5 = embeddings_2 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            hidden_states_6,
            (64,),
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_7 = torch._C._nn.linear(
            layer_norm_3,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_3 = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_5 = hidden_states_7.transpose(1, 2)
        hidden_states_7 = None
        hidden_states_8 = transpose_5.view(1, 256, 128, 128)
        transpose_5 = None
        hidden_states_9 = torch.conv2d(
            hidden_states_8,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        hidden_states_8 = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_1 = hidden_states_9.flatten(2)
        hidden_states_9 = None
        hidden_states_10 = flatten_1.transpose(1, 2)
        flatten_1 = None
        hidden_states_11 = torch._C._nn.gelu(hidden_states_10)
        hidden_states_10 = None
        hidden_states_12 = torch.nn.functional.dropout(
            hidden_states_11, 0.0, False, False
        )
        hidden_states_11 = None
        hidden_states_13 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_12 = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_14 = torch.nn.functional.dropout(
            hidden_states_13, 0.0, False, False
        )
        hidden_states_13 = None
        layer_output = hidden_states_14 + hidden_states_6
        hidden_states_14 = hidden_states_6 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            layer_output,
            (64,),
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_1_parameters_bias_ = (None)
        linear_6 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_5 = linear_6.view(1, -1, 1, 64)
        linear_6 = None
        query_layer_1 = view_5.transpose(1, 2)
        view_5 = None
        permute_3 = layer_norm_4.permute(0, 2, 1)
        layer_norm_4 = None
        hidden_states_15 = permute_3.reshape(1, 64, 128, 128)
        permute_3 = None
        hidden_states_16 = torch.conv2d(
            hidden_states_15,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_,
            (8, 8),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_15 = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_3 = hidden_states_16.reshape(1, 64, -1)
        hidden_states_16 = None
        hidden_states_17 = reshape_3.permute(0, 2, 1)
        reshape_3 = None
        hidden_states_18 = torch.nn.functional.layer_norm(
            hidden_states_17,
            (64,),
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_17 = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_7 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_6 = linear_7.view(1, -1, 1, 64)
        linear_7 = None
        key_layer_1 = view_6.transpose(1, 2)
        view_6 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_18 = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_7 = linear_8.view(1, -1, 1, 64)
        linear_8 = None
        value_layer_1 = view_7.transpose(1, 2)
        view_7 = None
        transpose_10 = key_layer_1.transpose(-1, -2)
        key_layer_1 = None
        attention_scores_2 = torch.matmul(query_layer_1, transpose_10)
        query_layer_1 = transpose_10 = None
        attention_scores_3 = attention_scores_2 / 8.0
        attention_scores_2 = None
        attention_probs_2 = torch.nn.functional.softmax(attention_scores_3, dim=-1)
        attention_scores_3 = None
        attention_probs_3 = torch.nn.functional.dropout(
            attention_probs_2, 0.0, False, False
        )
        attention_probs_2 = None
        context_layer_3 = torch.matmul(attention_probs_3, value_layer_1)
        attention_probs_3 = value_layer_1 = None
        permute_5 = context_layer_3.permute(0, 2, 1, 3)
        context_layer_3 = None
        context_layer_4 = permute_5.contiguous()
        permute_5 = None
        context_layer_5 = context_layer_4.view((1, 16384, 64))
        context_layer_4 = None
        hidden_states_19 = torch._C._nn.linear(
            context_layer_5,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_5 = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_20 = torch.nn.functional.dropout(
            hidden_states_19, 0.0, False, False
        )
        hidden_states_19 = None
        hidden_states_21 = hidden_states_20 + layer_output
        hidden_states_20 = layer_output = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            hidden_states_21,
            (64,),
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_22 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_6 = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_11 = hidden_states_22.transpose(1, 2)
        hidden_states_22 = None
        hidden_states_23 = transpose_11.view(1, 256, 128, 128)
        transpose_11 = None
        hidden_states_24 = torch.conv2d(
            hidden_states_23,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        hidden_states_23 = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_2 = hidden_states_24.flatten(2)
        hidden_states_24 = None
        hidden_states_25 = flatten_2.transpose(1, 2)
        flatten_2 = None
        hidden_states_26 = torch._C._nn.gelu(hidden_states_25)
        hidden_states_25 = None
        hidden_states_27 = torch.nn.functional.dropout(
            hidden_states_26, 0.0, False, False
        )
        hidden_states_26 = None
        hidden_states_28 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_27 = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_29 = torch.nn.functional.dropout(
            hidden_states_28, 0.0, False, False
        )
        hidden_states_28 = None
        layer_output_1 = hidden_states_29 + hidden_states_21
        hidden_states_29 = hidden_states_21 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            layer_output_1,
            (64,),
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_1_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_10 = linear_12.view(1, -1, 1, 64)
        linear_12 = None
        query_layer_2 = view_10.transpose(1, 2)
        view_10 = None
        permute_6 = layer_norm_7.permute(0, 2, 1)
        layer_norm_7 = None
        hidden_states_30 = permute_6.reshape(1, 64, 128, 128)
        permute_6 = None
        hidden_states_31 = torch.conv2d(
            hidden_states_30,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_,
            (8, 8),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_30 = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_5 = hidden_states_31.reshape(1, 64, -1)
        hidden_states_31 = None
        hidden_states_32 = reshape_5.permute(0, 2, 1)
        reshape_5 = None
        hidden_states_33 = torch.nn.functional.layer_norm(
            hidden_states_32,
            (64,),
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_32 = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_13 = torch._C._nn.linear(
            hidden_states_33,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_11 = linear_13.view(1, -1, 1, 64)
        linear_13 = None
        key_layer_2 = view_11.transpose(1, 2)
        view_11 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_33,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_33 = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_12 = linear_14.view(1, -1, 1, 64)
        linear_14 = None
        value_layer_2 = view_12.transpose(1, 2)
        view_12 = None
        transpose_16 = key_layer_2.transpose(-1, -2)
        key_layer_2 = None
        attention_scores_4 = torch.matmul(query_layer_2, transpose_16)
        query_layer_2 = transpose_16 = None
        attention_scores_5 = attention_scores_4 / 8.0
        attention_scores_4 = None
        attention_probs_4 = torch.nn.functional.softmax(attention_scores_5, dim=-1)
        attention_scores_5 = None
        attention_probs_5 = torch.nn.functional.dropout(
            attention_probs_4, 0.0, False, False
        )
        attention_probs_4 = None
        context_layer_6 = torch.matmul(attention_probs_5, value_layer_2)
        attention_probs_5 = value_layer_2 = None
        permute_8 = context_layer_6.permute(0, 2, 1, 3)
        context_layer_6 = None
        context_layer_7 = permute_8.contiguous()
        permute_8 = None
        context_layer_8 = context_layer_7.view((1, 16384, 64))
        context_layer_7 = None
        hidden_states_34 = torch._C._nn.linear(
            context_layer_8,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_8 = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_35 = torch.nn.functional.dropout(
            hidden_states_34, 0.0, False, False
        )
        hidden_states_34 = None
        hidden_states_36 = hidden_states_35 + layer_output_1
        hidden_states_35 = layer_output_1 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            hidden_states_36,
            (64,),
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_37 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_9 = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_17 = hidden_states_37.transpose(1, 2)
        hidden_states_37 = None
        hidden_states_38 = transpose_17.view(1, 256, 128, 128)
        transpose_17 = None
        hidden_states_39 = torch.conv2d(
            hidden_states_38,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        hidden_states_38 = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_3 = hidden_states_39.flatten(2)
        hidden_states_39 = None
        hidden_states_40 = flatten_3.transpose(1, 2)
        flatten_3 = None
        hidden_states_41 = torch._C._nn.gelu(hidden_states_40)
        hidden_states_40 = None
        hidden_states_42 = torch.nn.functional.dropout(
            hidden_states_41, 0.0, False, False
        )
        hidden_states_41 = None
        hidden_states_43 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_42 = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_0_modules_2_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_44 = torch.nn.functional.dropout(
            hidden_states_43, 0.0, False, False
        )
        hidden_states_43 = None
        layer_output_2 = hidden_states_44 + hidden_states_36
        hidden_states_44 = hidden_states_36 = None
        hidden_states_45 = torch.nn.functional.layer_norm(
            layer_output_2,
            (64,),
            l_self_modules_encoder_modules_layer_norm_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layer_norm_modules_0_parameters_bias_,
            1e-05,
        )
        layer_output_2 = (
            l_self_modules_encoder_modules_layer_norm_modules_0_parameters_weight_
        ) = l_self_modules_encoder_modules_layer_norm_modules_0_parameters_bias_ = None
        reshape_6 = hidden_states_45.reshape(1, 128, 128, -1)
        hidden_states_45 = None
        permute_9 = reshape_6.permute(0, 3, 1, 2)
        reshape_6 = None
        hidden_states_46 = permute_9.contiguous()
        permute_9 = None
        embeddings_3 = torch.conv2d(
            hidden_states_46,
            l_self_modules_encoder_modules_patch_embeddings_modules_1_modules_proj_parameters_weight_,
            l_self_modules_encoder_modules_patch_embeddings_modules_1_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_states_46 = l_self_modules_encoder_modules_patch_embeddings_modules_1_modules_proj_parameters_weight_ = l_self_modules_encoder_modules_patch_embeddings_modules_1_modules_proj_parameters_bias_ = (None)
        flatten_4 = embeddings_3.flatten(2)
        embeddings_3 = None
        embeddings_4 = flatten_4.transpose(1, 2)
        flatten_4 = None
        embeddings_5 = torch.nn.functional.layer_norm(
            embeddings_4,
            (128,),
            l_self_modules_encoder_modules_patch_embeddings_modules_1_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_patch_embeddings_modules_1_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        embeddings_4 = l_self_modules_encoder_modules_patch_embeddings_modules_1_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_patch_embeddings_modules_1_modules_layer_norm_parameters_bias_ = (None)
        layer_norm_12 = torch.nn.functional.layer_norm(
            embeddings_5,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_1_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_15 = linear_18.view(1, -1, 2, 64)
        linear_18 = None
        query_layer_3 = view_15.transpose(1, 2)
        view_15 = None
        permute_10 = layer_norm_12.permute(0, 2, 1)
        layer_norm_12 = None
        hidden_states_47 = permute_10.reshape(1, 128, 64, 64)
        permute_10 = None
        hidden_states_48 = torch.conv2d(
            hidden_states_47,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_47 = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_8 = hidden_states_48.reshape(1, 128, -1)
        hidden_states_48 = None
        hidden_states_49 = reshape_8.permute(0, 2, 1)
        reshape_8 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_49 = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_19 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_16 = linear_19.view(1, -1, 2, 64)
        linear_19 = None
        key_layer_3 = view_16.transpose(1, 2)
        view_16 = None
        linear_20 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_50 = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_17 = linear_20.view(1, -1, 2, 64)
        linear_20 = None
        value_layer_3 = view_17.transpose(1, 2)
        view_17 = None
        transpose_23 = key_layer_3.transpose(-1, -2)
        key_layer_3 = None
        attention_scores_6 = torch.matmul(query_layer_3, transpose_23)
        query_layer_3 = transpose_23 = None
        attention_scores_7 = attention_scores_6 / 8.0
        attention_scores_6 = None
        attention_probs_6 = torch.nn.functional.softmax(attention_scores_7, dim=-1)
        attention_scores_7 = None
        attention_probs_7 = torch.nn.functional.dropout(
            attention_probs_6, 0.0, False, False
        )
        attention_probs_6 = None
        context_layer_9 = torch.matmul(attention_probs_7, value_layer_3)
        attention_probs_7 = value_layer_3 = None
        permute_12 = context_layer_9.permute(0, 2, 1, 3)
        context_layer_9 = None
        context_layer_10 = permute_12.contiguous()
        permute_12 = None
        context_layer_11 = context_layer_10.view((1, 4096, 128))
        context_layer_10 = None
        hidden_states_51 = torch._C._nn.linear(
            context_layer_11,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_11 = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_52 = torch.nn.functional.dropout(
            hidden_states_51, 0.0, False, False
        )
        hidden_states_51 = None
        hidden_states_53 = hidden_states_52 + embeddings_5
        hidden_states_52 = embeddings_5 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            hidden_states_53,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_54 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_14 = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_24 = hidden_states_54.transpose(1, 2)
        hidden_states_54 = None
        hidden_states_55 = transpose_24.view(1, 512, 64, 64)
        transpose_24 = None
        hidden_states_56 = torch.conv2d(
            hidden_states_55,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        hidden_states_55 = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_5 = hidden_states_56.flatten(2)
        hidden_states_56 = None
        hidden_states_57 = flatten_5.transpose(1, 2)
        flatten_5 = None
        hidden_states_58 = torch._C._nn.gelu(hidden_states_57)
        hidden_states_57 = None
        hidden_states_59 = torch.nn.functional.dropout(
            hidden_states_58, 0.0, False, False
        )
        hidden_states_58 = None
        hidden_states_60 = torch._C._nn.linear(
            hidden_states_59,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_59 = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_61 = torch.nn.functional.dropout(
            hidden_states_60, 0.0, False, False
        )
        hidden_states_60 = None
        layer_output_3 = hidden_states_61 + hidden_states_53
        hidden_states_61 = hidden_states_53 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            layer_output_3,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_1_parameters_bias_ = (None)
        linear_24 = torch._C._nn.linear(
            layer_norm_15,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_20 = linear_24.view(1, -1, 2, 64)
        linear_24 = None
        query_layer_4 = view_20.transpose(1, 2)
        view_20 = None
        permute_13 = layer_norm_15.permute(0, 2, 1)
        layer_norm_15 = None
        hidden_states_62 = permute_13.reshape(1, 128, 64, 64)
        permute_13 = None
        hidden_states_63 = torch.conv2d(
            hidden_states_62,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_62 = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_10 = hidden_states_63.reshape(1, 128, -1)
        hidden_states_63 = None
        hidden_states_64 = reshape_10.permute(0, 2, 1)
        reshape_10 = None
        hidden_states_65 = torch.nn.functional.layer_norm(
            hidden_states_64,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_64 = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_25 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_21 = linear_25.view(1, -1, 2, 64)
        linear_25 = None
        key_layer_4 = view_21.transpose(1, 2)
        view_21 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_65 = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_22 = linear_26.view(1, -1, 2, 64)
        linear_26 = None
        value_layer_4 = view_22.transpose(1, 2)
        view_22 = None
        transpose_29 = key_layer_4.transpose(-1, -2)
        key_layer_4 = None
        attention_scores_8 = torch.matmul(query_layer_4, transpose_29)
        query_layer_4 = transpose_29 = None
        attention_scores_9 = attention_scores_8 / 8.0
        attention_scores_8 = None
        attention_probs_8 = torch.nn.functional.softmax(attention_scores_9, dim=-1)
        attention_scores_9 = None
        attention_probs_9 = torch.nn.functional.dropout(
            attention_probs_8, 0.0, False, False
        )
        attention_probs_8 = None
        context_layer_12 = torch.matmul(attention_probs_9, value_layer_4)
        attention_probs_9 = value_layer_4 = None
        permute_15 = context_layer_12.permute(0, 2, 1, 3)
        context_layer_12 = None
        context_layer_13 = permute_15.contiguous()
        permute_15 = None
        context_layer_14 = context_layer_13.view((1, 4096, 128))
        context_layer_13 = None
        hidden_states_66 = torch._C._nn.linear(
            context_layer_14,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_14 = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_67 = torch.nn.functional.dropout(
            hidden_states_66, 0.0, False, False
        )
        hidden_states_66 = None
        hidden_states_68 = hidden_states_67 + layer_output_3
        hidden_states_67 = layer_output_3 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            hidden_states_68,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_69 = torch._C._nn.linear(
            layer_norm_17,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_17 = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_30 = hidden_states_69.transpose(1, 2)
        hidden_states_69 = None
        hidden_states_70 = transpose_30.view(1, 512, 64, 64)
        transpose_30 = None
        hidden_states_71 = torch.conv2d(
            hidden_states_70,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        hidden_states_70 = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_6 = hidden_states_71.flatten(2)
        hidden_states_71 = None
        hidden_states_72 = flatten_6.transpose(1, 2)
        flatten_6 = None
        hidden_states_73 = torch._C._nn.gelu(hidden_states_72)
        hidden_states_72 = None
        hidden_states_74 = torch.nn.functional.dropout(
            hidden_states_73, 0.0, False, False
        )
        hidden_states_73 = None
        hidden_states_75 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_74 = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_76 = torch.nn.functional.dropout(
            hidden_states_75, 0.0, False, False
        )
        hidden_states_75 = None
        layer_output_4 = hidden_states_76 + hidden_states_68
        hidden_states_76 = hidden_states_68 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            layer_output_4,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_1_parameters_bias_ = (None)
        linear_30 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_25 = linear_30.view(1, -1, 2, 64)
        linear_30 = None
        query_layer_5 = view_25.transpose(1, 2)
        view_25 = None
        permute_16 = layer_norm_18.permute(0, 2, 1)
        layer_norm_18 = None
        hidden_states_77 = permute_16.reshape(1, 128, 64, 64)
        permute_16 = None
        hidden_states_78 = torch.conv2d(
            hidden_states_77,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_77 = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_12 = hidden_states_78.reshape(1, 128, -1)
        hidden_states_78 = None
        hidden_states_79 = reshape_12.permute(0, 2, 1)
        reshape_12 = None
        hidden_states_80 = torch.nn.functional.layer_norm(
            hidden_states_79,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_79 = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_31 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_26 = linear_31.view(1, -1, 2, 64)
        linear_31 = None
        key_layer_5 = view_26.transpose(1, 2)
        view_26 = None
        linear_32 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_80 = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_27 = linear_32.view(1, -1, 2, 64)
        linear_32 = None
        value_layer_5 = view_27.transpose(1, 2)
        view_27 = None
        transpose_35 = key_layer_5.transpose(-1, -2)
        key_layer_5 = None
        attention_scores_10 = torch.matmul(query_layer_5, transpose_35)
        query_layer_5 = transpose_35 = None
        attention_scores_11 = attention_scores_10 / 8.0
        attention_scores_10 = None
        attention_probs_10 = torch.nn.functional.softmax(attention_scores_11, dim=-1)
        attention_scores_11 = None
        attention_probs_11 = torch.nn.functional.dropout(
            attention_probs_10, 0.0, False, False
        )
        attention_probs_10 = None
        context_layer_15 = torch.matmul(attention_probs_11, value_layer_5)
        attention_probs_11 = value_layer_5 = None
        permute_18 = context_layer_15.permute(0, 2, 1, 3)
        context_layer_15 = None
        context_layer_16 = permute_18.contiguous()
        permute_18 = None
        context_layer_17 = context_layer_16.view((1, 4096, 128))
        context_layer_16 = None
        hidden_states_81 = torch._C._nn.linear(
            context_layer_17,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_17 = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_82 = torch.nn.functional.dropout(
            hidden_states_81, 0.0, False, False
        )
        hidden_states_81 = None
        hidden_states_83 = hidden_states_82 + layer_output_4
        hidden_states_82 = layer_output_4 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            hidden_states_83,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_84 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_20 = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_36 = hidden_states_84.transpose(1, 2)
        hidden_states_84 = None
        hidden_states_85 = transpose_36.view(1, 512, 64, 64)
        transpose_36 = None
        hidden_states_86 = torch.conv2d(
            hidden_states_85,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        hidden_states_85 = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_7 = hidden_states_86.flatten(2)
        hidden_states_86 = None
        hidden_states_87 = flatten_7.transpose(1, 2)
        flatten_7 = None
        hidden_states_88 = torch._C._nn.gelu(hidden_states_87)
        hidden_states_87 = None
        hidden_states_89 = torch.nn.functional.dropout(
            hidden_states_88, 0.0, False, False
        )
        hidden_states_88 = None
        hidden_states_90 = torch._C._nn.linear(
            hidden_states_89,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_89 = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_2_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_91 = torch.nn.functional.dropout(
            hidden_states_90, 0.0, False, False
        )
        hidden_states_90 = None
        layer_output_5 = hidden_states_91 + hidden_states_83
        hidden_states_91 = hidden_states_83 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            layer_output_5,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_1_parameters_bias_ = (None)
        linear_36 = torch._C._nn.linear(
            layer_norm_21,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_30 = linear_36.view(1, -1, 2, 64)
        linear_36 = None
        query_layer_6 = view_30.transpose(1, 2)
        view_30 = None
        permute_19 = layer_norm_21.permute(0, 2, 1)
        layer_norm_21 = None
        hidden_states_92 = permute_19.reshape(1, 128, 64, 64)
        permute_19 = None
        hidden_states_93 = torch.conv2d(
            hidden_states_92,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_92 = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_14 = hidden_states_93.reshape(1, 128, -1)
        hidden_states_93 = None
        hidden_states_94 = reshape_14.permute(0, 2, 1)
        reshape_14 = None
        hidden_states_95 = torch.nn.functional.layer_norm(
            hidden_states_94,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_94 = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_37 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_31 = linear_37.view(1, -1, 2, 64)
        linear_37 = None
        key_layer_6 = view_31.transpose(1, 2)
        view_31 = None
        linear_38 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_95 = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_32 = linear_38.view(1, -1, 2, 64)
        linear_38 = None
        value_layer_6 = view_32.transpose(1, 2)
        view_32 = None
        transpose_41 = key_layer_6.transpose(-1, -2)
        key_layer_6 = None
        attention_scores_12 = torch.matmul(query_layer_6, transpose_41)
        query_layer_6 = transpose_41 = None
        attention_scores_13 = attention_scores_12 / 8.0
        attention_scores_12 = None
        attention_probs_12 = torch.nn.functional.softmax(attention_scores_13, dim=-1)
        attention_scores_13 = None
        attention_probs_13 = torch.nn.functional.dropout(
            attention_probs_12, 0.0, False, False
        )
        attention_probs_12 = None
        context_layer_18 = torch.matmul(attention_probs_13, value_layer_6)
        attention_probs_13 = value_layer_6 = None
        permute_21 = context_layer_18.permute(0, 2, 1, 3)
        context_layer_18 = None
        context_layer_19 = permute_21.contiguous()
        permute_21 = None
        context_layer_20 = context_layer_19.view((1, 4096, 128))
        context_layer_19 = None
        hidden_states_96 = torch._C._nn.linear(
            context_layer_20,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_20 = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_97 = torch.nn.functional.dropout(
            hidden_states_96, 0.0, False, False
        )
        hidden_states_96 = None
        hidden_states_98 = hidden_states_97 + layer_output_5
        hidden_states_97 = layer_output_5 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            hidden_states_98,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_99 = torch._C._nn.linear(
            layer_norm_23,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_23 = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_42 = hidden_states_99.transpose(1, 2)
        hidden_states_99 = None
        hidden_states_100 = transpose_42.view(1, 512, 64, 64)
        transpose_42 = None
        hidden_states_101 = torch.conv2d(
            hidden_states_100,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        hidden_states_100 = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_8 = hidden_states_101.flatten(2)
        hidden_states_101 = None
        hidden_states_102 = flatten_8.transpose(1, 2)
        flatten_8 = None
        hidden_states_103 = torch._C._nn.gelu(hidden_states_102)
        hidden_states_102 = None
        hidden_states_104 = torch.nn.functional.dropout(
            hidden_states_103, 0.0, False, False
        )
        hidden_states_103 = None
        hidden_states_105 = torch._C._nn.linear(
            hidden_states_104,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_104 = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_3_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_106 = torch.nn.functional.dropout(
            hidden_states_105, 0.0, False, False
        )
        hidden_states_105 = None
        layer_output_6 = hidden_states_106 + hidden_states_98
        hidden_states_106 = hidden_states_98 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            layer_output_6,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_1_parameters_bias_ = (None)
        linear_42 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_35 = linear_42.view(1, -1, 2, 64)
        linear_42 = None
        query_layer_7 = view_35.transpose(1, 2)
        view_35 = None
        permute_22 = layer_norm_24.permute(0, 2, 1)
        layer_norm_24 = None
        hidden_states_107 = permute_22.reshape(1, 128, 64, 64)
        permute_22 = None
        hidden_states_108 = torch.conv2d(
            hidden_states_107,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_107 = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_16 = hidden_states_108.reshape(1, 128, -1)
        hidden_states_108 = None
        hidden_states_109 = reshape_16.permute(0, 2, 1)
        reshape_16 = None
        hidden_states_110 = torch.nn.functional.layer_norm(
            hidden_states_109,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_109 = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_43 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_36 = linear_43.view(1, -1, 2, 64)
        linear_43 = None
        key_layer_7 = view_36.transpose(1, 2)
        view_36 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_110 = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_37 = linear_44.view(1, -1, 2, 64)
        linear_44 = None
        value_layer_7 = view_37.transpose(1, 2)
        view_37 = None
        transpose_47 = key_layer_7.transpose(-1, -2)
        key_layer_7 = None
        attention_scores_14 = torch.matmul(query_layer_7, transpose_47)
        query_layer_7 = transpose_47 = None
        attention_scores_15 = attention_scores_14 / 8.0
        attention_scores_14 = None
        attention_probs_14 = torch.nn.functional.softmax(attention_scores_15, dim=-1)
        attention_scores_15 = None
        attention_probs_15 = torch.nn.functional.dropout(
            attention_probs_14, 0.0, False, False
        )
        attention_probs_14 = None
        context_layer_21 = torch.matmul(attention_probs_15, value_layer_7)
        attention_probs_15 = value_layer_7 = None
        permute_24 = context_layer_21.permute(0, 2, 1, 3)
        context_layer_21 = None
        context_layer_22 = permute_24.contiguous()
        permute_24 = None
        context_layer_23 = context_layer_22.view((1, 4096, 128))
        context_layer_22 = None
        hidden_states_111 = torch._C._nn.linear(
            context_layer_23,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_23 = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_112 = torch.nn.functional.dropout(
            hidden_states_111, 0.0, False, False
        )
        hidden_states_111 = None
        hidden_states_113 = hidden_states_112 + layer_output_6
        hidden_states_112 = layer_output_6 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            hidden_states_113,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_114 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_26 = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_48 = hidden_states_114.transpose(1, 2)
        hidden_states_114 = None
        hidden_states_115 = transpose_48.view(1, 512, 64, 64)
        transpose_48 = None
        hidden_states_116 = torch.conv2d(
            hidden_states_115,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        hidden_states_115 = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_9 = hidden_states_116.flatten(2)
        hidden_states_116 = None
        hidden_states_117 = flatten_9.transpose(1, 2)
        flatten_9 = None
        hidden_states_118 = torch._C._nn.gelu(hidden_states_117)
        hidden_states_117 = None
        hidden_states_119 = torch.nn.functional.dropout(
            hidden_states_118, 0.0, False, False
        )
        hidden_states_118 = None
        hidden_states_120 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_119 = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_4_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_121 = torch.nn.functional.dropout(
            hidden_states_120, 0.0, False, False
        )
        hidden_states_120 = None
        layer_output_7 = hidden_states_121 + hidden_states_113
        hidden_states_121 = hidden_states_113 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            layer_output_7,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_1_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            layer_norm_27,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_40 = linear_48.view(1, -1, 2, 64)
        linear_48 = None
        query_layer_8 = view_40.transpose(1, 2)
        view_40 = None
        permute_25 = layer_norm_27.permute(0, 2, 1)
        layer_norm_27 = None
        hidden_states_122 = permute_25.reshape(1, 128, 64, 64)
        permute_25 = None
        hidden_states_123 = torch.conv2d(
            hidden_states_122,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_122 = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_18 = hidden_states_123.reshape(1, 128, -1)
        hidden_states_123 = None
        hidden_states_124 = reshape_18.permute(0, 2, 1)
        reshape_18 = None
        hidden_states_125 = torch.nn.functional.layer_norm(
            hidden_states_124,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_124 = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_49 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_41 = linear_49.view(1, -1, 2, 64)
        linear_49 = None
        key_layer_8 = view_41.transpose(1, 2)
        view_41 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_125 = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_42 = linear_50.view(1, -1, 2, 64)
        linear_50 = None
        value_layer_8 = view_42.transpose(1, 2)
        view_42 = None
        transpose_53 = key_layer_8.transpose(-1, -2)
        key_layer_8 = None
        attention_scores_16 = torch.matmul(query_layer_8, transpose_53)
        query_layer_8 = transpose_53 = None
        attention_scores_17 = attention_scores_16 / 8.0
        attention_scores_16 = None
        attention_probs_16 = torch.nn.functional.softmax(attention_scores_17, dim=-1)
        attention_scores_17 = None
        attention_probs_17 = torch.nn.functional.dropout(
            attention_probs_16, 0.0, False, False
        )
        attention_probs_16 = None
        context_layer_24 = torch.matmul(attention_probs_17, value_layer_8)
        attention_probs_17 = value_layer_8 = None
        permute_27 = context_layer_24.permute(0, 2, 1, 3)
        context_layer_24 = None
        context_layer_25 = permute_27.contiguous()
        permute_27 = None
        context_layer_26 = context_layer_25.view((1, 4096, 128))
        context_layer_25 = None
        hidden_states_126 = torch._C._nn.linear(
            context_layer_26,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_26 = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_127 = torch.nn.functional.dropout(
            hidden_states_126, 0.0, False, False
        )
        hidden_states_126 = None
        hidden_states_128 = hidden_states_127 + layer_output_7
        hidden_states_127 = layer_output_7 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            hidden_states_128,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_129 = torch._C._nn.linear(
            layer_norm_29,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_29 = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_54 = hidden_states_129.transpose(1, 2)
        hidden_states_129 = None
        hidden_states_130 = transpose_54.view(1, 512, 64, 64)
        transpose_54 = None
        hidden_states_131 = torch.conv2d(
            hidden_states_130,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        hidden_states_130 = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_10 = hidden_states_131.flatten(2)
        hidden_states_131 = None
        hidden_states_132 = flatten_10.transpose(1, 2)
        flatten_10 = None
        hidden_states_133 = torch._C._nn.gelu(hidden_states_132)
        hidden_states_132 = None
        hidden_states_134 = torch.nn.functional.dropout(
            hidden_states_133, 0.0, False, False
        )
        hidden_states_133 = None
        hidden_states_135 = torch._C._nn.linear(
            hidden_states_134,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_134 = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_5_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_136 = torch.nn.functional.dropout(
            hidden_states_135, 0.0, False, False
        )
        hidden_states_135 = None
        layer_output_8 = hidden_states_136 + hidden_states_128
        hidden_states_136 = hidden_states_128 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            layer_output_8,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_1_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_45 = linear_54.view(1, -1, 2, 64)
        linear_54 = None
        query_layer_9 = view_45.transpose(1, 2)
        view_45 = None
        permute_28 = layer_norm_30.permute(0, 2, 1)
        layer_norm_30 = None
        hidden_states_137 = permute_28.reshape(1, 128, 64, 64)
        permute_28 = None
        hidden_states_138 = torch.conv2d(
            hidden_states_137,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_137 = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_20 = hidden_states_138.reshape(1, 128, -1)
        hidden_states_138 = None
        hidden_states_139 = reshape_20.permute(0, 2, 1)
        reshape_20 = None
        hidden_states_140 = torch.nn.functional.layer_norm(
            hidden_states_139,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_139 = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_55 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_46 = linear_55.view(1, -1, 2, 64)
        linear_55 = None
        key_layer_9 = view_46.transpose(1, 2)
        view_46 = None
        linear_56 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_140 = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_47 = linear_56.view(1, -1, 2, 64)
        linear_56 = None
        value_layer_9 = view_47.transpose(1, 2)
        view_47 = None
        transpose_59 = key_layer_9.transpose(-1, -2)
        key_layer_9 = None
        attention_scores_18 = torch.matmul(query_layer_9, transpose_59)
        query_layer_9 = transpose_59 = None
        attention_scores_19 = attention_scores_18 / 8.0
        attention_scores_18 = None
        attention_probs_18 = torch.nn.functional.softmax(attention_scores_19, dim=-1)
        attention_scores_19 = None
        attention_probs_19 = torch.nn.functional.dropout(
            attention_probs_18, 0.0, False, False
        )
        attention_probs_18 = None
        context_layer_27 = torch.matmul(attention_probs_19, value_layer_9)
        attention_probs_19 = value_layer_9 = None
        permute_30 = context_layer_27.permute(0, 2, 1, 3)
        context_layer_27 = None
        context_layer_28 = permute_30.contiguous()
        permute_30 = None
        context_layer_29 = context_layer_28.view((1, 4096, 128))
        context_layer_28 = None
        hidden_states_141 = torch._C._nn.linear(
            context_layer_29,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_29 = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_142 = torch.nn.functional.dropout(
            hidden_states_141, 0.0, False, False
        )
        hidden_states_141 = None
        hidden_states_143 = hidden_states_142 + layer_output_8
        hidden_states_142 = layer_output_8 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            hidden_states_143,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_144 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_32 = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_60 = hidden_states_144.transpose(1, 2)
        hidden_states_144 = None
        hidden_states_145 = transpose_60.view(1, 512, 64, 64)
        transpose_60 = None
        hidden_states_146 = torch.conv2d(
            hidden_states_145,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        hidden_states_145 = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_11 = hidden_states_146.flatten(2)
        hidden_states_146 = None
        hidden_states_147 = flatten_11.transpose(1, 2)
        flatten_11 = None
        hidden_states_148 = torch._C._nn.gelu(hidden_states_147)
        hidden_states_147 = None
        hidden_states_149 = torch.nn.functional.dropout(
            hidden_states_148, 0.0, False, False
        )
        hidden_states_148 = None
        hidden_states_150 = torch._C._nn.linear(
            hidden_states_149,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_149 = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_6_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_151 = torch.nn.functional.dropout(
            hidden_states_150, 0.0, False, False
        )
        hidden_states_150 = None
        layer_output_9 = hidden_states_151 + hidden_states_143
        hidden_states_151 = hidden_states_143 = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            layer_output_9,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_1_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            layer_norm_33,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_50 = linear_60.view(1, -1, 2, 64)
        linear_60 = None
        query_layer_10 = view_50.transpose(1, 2)
        view_50 = None
        permute_31 = layer_norm_33.permute(0, 2, 1)
        layer_norm_33 = None
        hidden_states_152 = permute_31.reshape(1, 128, 64, 64)
        permute_31 = None
        hidden_states_153 = torch.conv2d(
            hidden_states_152,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_sr_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_152 = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_22 = hidden_states_153.reshape(1, 128, -1)
        hidden_states_153 = None
        hidden_states_154 = reshape_22.permute(0, 2, 1)
        reshape_22 = None
        hidden_states_155 = torch.nn.functional.layer_norm(
            hidden_states_154,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_154 = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_61 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_51 = linear_61.view(1, -1, 2, 64)
        linear_61 = None
        key_layer_10 = view_51.transpose(1, 2)
        view_51 = None
        linear_62 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_155 = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_52 = linear_62.view(1, -1, 2, 64)
        linear_62 = None
        value_layer_10 = view_52.transpose(1, 2)
        view_52 = None
        transpose_65 = key_layer_10.transpose(-1, -2)
        key_layer_10 = None
        attention_scores_20 = torch.matmul(query_layer_10, transpose_65)
        query_layer_10 = transpose_65 = None
        attention_scores_21 = attention_scores_20 / 8.0
        attention_scores_20 = None
        attention_probs_20 = torch.nn.functional.softmax(attention_scores_21, dim=-1)
        attention_scores_21 = None
        attention_probs_21 = torch.nn.functional.dropout(
            attention_probs_20, 0.0, False, False
        )
        attention_probs_20 = None
        context_layer_30 = torch.matmul(attention_probs_21, value_layer_10)
        attention_probs_21 = value_layer_10 = None
        permute_33 = context_layer_30.permute(0, 2, 1, 3)
        context_layer_30 = None
        context_layer_31 = permute_33.contiguous()
        permute_33 = None
        context_layer_32 = context_layer_31.view((1, 4096, 128))
        context_layer_31 = None
        hidden_states_156 = torch._C._nn.linear(
            context_layer_32,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_32 = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_157 = torch.nn.functional.dropout(
            hidden_states_156, 0.0, False, False
        )
        hidden_states_156 = None
        hidden_states_158 = hidden_states_157 + layer_output_9
        hidden_states_157 = layer_output_9 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            hidden_states_158,
            (128,),
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_159 = torch._C._nn.linear(
            layer_norm_35,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_35 = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_66 = hidden_states_159.transpose(1, 2)
        hidden_states_159 = None
        hidden_states_160 = transpose_66.view(1, 512, 64, 64)
        transpose_66 = None
        hidden_states_161 = torch.conv2d(
            hidden_states_160,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        hidden_states_160 = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_12 = hidden_states_161.flatten(2)
        hidden_states_161 = None
        hidden_states_162 = flatten_12.transpose(1, 2)
        flatten_12 = None
        hidden_states_163 = torch._C._nn.gelu(hidden_states_162)
        hidden_states_162 = None
        hidden_states_164 = torch.nn.functional.dropout(
            hidden_states_163, 0.0, False, False
        )
        hidden_states_163 = None
        hidden_states_165 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_164 = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_1_modules_7_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_166 = torch.nn.functional.dropout(
            hidden_states_165, 0.0, False, False
        )
        hidden_states_165 = None
        layer_output_10 = hidden_states_166 + hidden_states_158
        hidden_states_166 = hidden_states_158 = None
        hidden_states_167 = torch.nn.functional.layer_norm(
            layer_output_10,
            (128,),
            l_self_modules_encoder_modules_layer_norm_modules_1_parameters_weight_,
            l_self_modules_encoder_modules_layer_norm_modules_1_parameters_bias_,
            1e-05,
        )
        layer_output_10 = (
            l_self_modules_encoder_modules_layer_norm_modules_1_parameters_weight_
        ) = l_self_modules_encoder_modules_layer_norm_modules_1_parameters_bias_ = None
        reshape_23 = hidden_states_167.reshape(1, 64, 64, -1)
        hidden_states_167 = None
        permute_34 = reshape_23.permute(0, 3, 1, 2)
        reshape_23 = None
        hidden_states_168 = permute_34.contiguous()
        permute_34 = None
        embeddings_6 = torch.conv2d(
            hidden_states_168,
            l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_weight_,
            l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_states_168 = l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_weight_ = l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_bias_ = (None)
        flatten_13 = embeddings_6.flatten(2)
        embeddings_6 = None
        embeddings_7 = flatten_13.transpose(1, 2)
        flatten_13 = None
        embeddings_8 = torch.nn.functional.layer_norm(
            embeddings_7,
            (320,),
            l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        embeddings_7 = l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_bias_ = (None)
        layer_norm_38 = torch.nn.functional.layer_norm(
            embeddings_8,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_bias_ = (None)
        linear_66 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_55 = linear_66.view(1, -1, 5, 64)
        linear_66 = None
        query_layer_11 = view_55.transpose(1, 2)
        view_55 = None
        permute_35 = layer_norm_38.permute(0, 2, 1)
        layer_norm_38 = None
        hidden_states_169 = permute_35.reshape(1, 320, 32, 32)
        permute_35 = None
        hidden_states_170 = torch.conv2d(
            hidden_states_169,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_169 = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_25 = hidden_states_170.reshape(1, 320, -1)
        hidden_states_170 = None
        hidden_states_171 = reshape_25.permute(0, 2, 1)
        reshape_25 = None
        hidden_states_172 = torch.nn.functional.layer_norm(
            hidden_states_171,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_171 = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_67 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_56 = linear_67.view(1, -1, 5, 64)
        linear_67 = None
        key_layer_11 = view_56.transpose(1, 2)
        view_56 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_172 = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_57 = linear_68.view(1, -1, 5, 64)
        linear_68 = None
        value_layer_11 = view_57.transpose(1, 2)
        view_57 = None
        transpose_72 = key_layer_11.transpose(-1, -2)
        key_layer_11 = None
        attention_scores_22 = torch.matmul(query_layer_11, transpose_72)
        query_layer_11 = transpose_72 = None
        attention_scores_23 = attention_scores_22 / 8.0
        attention_scores_22 = None
        attention_probs_22 = torch.nn.functional.softmax(attention_scores_23, dim=-1)
        attention_scores_23 = None
        attention_probs_23 = torch.nn.functional.dropout(
            attention_probs_22, 0.0, False, False
        )
        attention_probs_22 = None
        context_layer_33 = torch.matmul(attention_probs_23, value_layer_11)
        attention_probs_23 = value_layer_11 = None
        permute_37 = context_layer_33.permute(0, 2, 1, 3)
        context_layer_33 = None
        context_layer_34 = permute_37.contiguous()
        permute_37 = None
        context_layer_35 = context_layer_34.view((1, 1024, 320))
        context_layer_34 = None
        hidden_states_173 = torch._C._nn.linear(
            context_layer_35,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_35 = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_174 = torch.nn.functional.dropout(
            hidden_states_173, 0.0, False, False
        )
        hidden_states_173 = None
        hidden_states_175 = hidden_states_174 + embeddings_8
        hidden_states_174 = embeddings_8 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            hidden_states_175,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_176 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_40 = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_73 = hidden_states_176.transpose(1, 2)
        hidden_states_176 = None
        hidden_states_177 = transpose_73.view(1, 1280, 32, 32)
        transpose_73 = None
        hidden_states_178 = torch.conv2d(
            hidden_states_177,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_177 = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_14 = hidden_states_178.flatten(2)
        hidden_states_178 = None
        hidden_states_179 = flatten_14.transpose(1, 2)
        flatten_14 = None
        hidden_states_180 = torch._C._nn.gelu(hidden_states_179)
        hidden_states_179 = None
        hidden_states_181 = torch.nn.functional.dropout(
            hidden_states_180, 0.0, False, False
        )
        hidden_states_180 = None
        hidden_states_182 = torch._C._nn.linear(
            hidden_states_181,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_181 = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_183 = torch.nn.functional.dropout(
            hidden_states_182, 0.0, False, False
        )
        hidden_states_182 = None
        layer_output_11 = hidden_states_183 + hidden_states_175
        hidden_states_183 = hidden_states_175 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            layer_output_11,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_bias_ = (None)
        linear_72 = torch._C._nn.linear(
            layer_norm_41,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_60 = linear_72.view(1, -1, 5, 64)
        linear_72 = None
        query_layer_12 = view_60.transpose(1, 2)
        view_60 = None
        permute_38 = layer_norm_41.permute(0, 2, 1)
        layer_norm_41 = None
        hidden_states_184 = permute_38.reshape(1, 320, 32, 32)
        permute_38 = None
        hidden_states_185 = torch.conv2d(
            hidden_states_184,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_184 = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_27 = hidden_states_185.reshape(1, 320, -1)
        hidden_states_185 = None
        hidden_states_186 = reshape_27.permute(0, 2, 1)
        reshape_27 = None
        hidden_states_187 = torch.nn.functional.layer_norm(
            hidden_states_186,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_186 = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_73 = torch._C._nn.linear(
            hidden_states_187,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_61 = linear_73.view(1, -1, 5, 64)
        linear_73 = None
        key_layer_12 = view_61.transpose(1, 2)
        view_61 = None
        linear_74 = torch._C._nn.linear(
            hidden_states_187,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_187 = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_62 = linear_74.view(1, -1, 5, 64)
        linear_74 = None
        value_layer_12 = view_62.transpose(1, 2)
        view_62 = None
        transpose_78 = key_layer_12.transpose(-1, -2)
        key_layer_12 = None
        attention_scores_24 = torch.matmul(query_layer_12, transpose_78)
        query_layer_12 = transpose_78 = None
        attention_scores_25 = attention_scores_24 / 8.0
        attention_scores_24 = None
        attention_probs_24 = torch.nn.functional.softmax(attention_scores_25, dim=-1)
        attention_scores_25 = None
        attention_probs_25 = torch.nn.functional.dropout(
            attention_probs_24, 0.0, False, False
        )
        attention_probs_24 = None
        context_layer_36 = torch.matmul(attention_probs_25, value_layer_12)
        attention_probs_25 = value_layer_12 = None
        permute_40 = context_layer_36.permute(0, 2, 1, 3)
        context_layer_36 = None
        context_layer_37 = permute_40.contiguous()
        permute_40 = None
        context_layer_38 = context_layer_37.view((1, 1024, 320))
        context_layer_37 = None
        hidden_states_188 = torch._C._nn.linear(
            context_layer_38,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_38 = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_189 = torch.nn.functional.dropout(
            hidden_states_188, 0.0, False, False
        )
        hidden_states_188 = None
        hidden_states_190 = hidden_states_189 + layer_output_11
        hidden_states_189 = layer_output_11 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            hidden_states_190,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_191 = torch._C._nn.linear(
            layer_norm_43,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_43 = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_79 = hidden_states_191.transpose(1, 2)
        hidden_states_191 = None
        hidden_states_192 = transpose_79.view(1, 1280, 32, 32)
        transpose_79 = None
        hidden_states_193 = torch.conv2d(
            hidden_states_192,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_192 = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_15 = hidden_states_193.flatten(2)
        hidden_states_193 = None
        hidden_states_194 = flatten_15.transpose(1, 2)
        flatten_15 = None
        hidden_states_195 = torch._C._nn.gelu(hidden_states_194)
        hidden_states_194 = None
        hidden_states_196 = torch.nn.functional.dropout(
            hidden_states_195, 0.0, False, False
        )
        hidden_states_195 = None
        hidden_states_197 = torch._C._nn.linear(
            hidden_states_196,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_196 = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_198 = torch.nn.functional.dropout(
            hidden_states_197, 0.0, False, False
        )
        hidden_states_197 = None
        layer_output_12 = hidden_states_198 + hidden_states_190
        hidden_states_198 = hidden_states_190 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            layer_output_12,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_1_parameters_bias_ = (None)
        linear_78 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_65 = linear_78.view(1, -1, 5, 64)
        linear_78 = None
        query_layer_13 = view_65.transpose(1, 2)
        view_65 = None
        permute_41 = layer_norm_44.permute(0, 2, 1)
        layer_norm_44 = None
        hidden_states_199 = permute_41.reshape(1, 320, 32, 32)
        permute_41 = None
        hidden_states_200 = torch.conv2d(
            hidden_states_199,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_199 = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_29 = hidden_states_200.reshape(1, 320, -1)
        hidden_states_200 = None
        hidden_states_201 = reshape_29.permute(0, 2, 1)
        reshape_29 = None
        hidden_states_202 = torch.nn.functional.layer_norm(
            hidden_states_201,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_201 = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_79 = torch._C._nn.linear(
            hidden_states_202,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_66 = linear_79.view(1, -1, 5, 64)
        linear_79 = None
        key_layer_13 = view_66.transpose(1, 2)
        view_66 = None
        linear_80 = torch._C._nn.linear(
            hidden_states_202,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_202 = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_67 = linear_80.view(1, -1, 5, 64)
        linear_80 = None
        value_layer_13 = view_67.transpose(1, 2)
        view_67 = None
        transpose_84 = key_layer_13.transpose(-1, -2)
        key_layer_13 = None
        attention_scores_26 = torch.matmul(query_layer_13, transpose_84)
        query_layer_13 = transpose_84 = None
        attention_scores_27 = attention_scores_26 / 8.0
        attention_scores_26 = None
        attention_probs_26 = torch.nn.functional.softmax(attention_scores_27, dim=-1)
        attention_scores_27 = None
        attention_probs_27 = torch.nn.functional.dropout(
            attention_probs_26, 0.0, False, False
        )
        attention_probs_26 = None
        context_layer_39 = torch.matmul(attention_probs_27, value_layer_13)
        attention_probs_27 = value_layer_13 = None
        permute_43 = context_layer_39.permute(0, 2, 1, 3)
        context_layer_39 = None
        context_layer_40 = permute_43.contiguous()
        permute_43 = None
        context_layer_41 = context_layer_40.view((1, 1024, 320))
        context_layer_40 = None
        hidden_states_203 = torch._C._nn.linear(
            context_layer_41,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_41 = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_204 = torch.nn.functional.dropout(
            hidden_states_203, 0.0, False, False
        )
        hidden_states_203 = None
        hidden_states_205 = hidden_states_204 + layer_output_12
        hidden_states_204 = layer_output_12 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            hidden_states_205,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_206 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_46 = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_85 = hidden_states_206.transpose(1, 2)
        hidden_states_206 = None
        hidden_states_207 = transpose_85.view(1, 1280, 32, 32)
        transpose_85 = None
        hidden_states_208 = torch.conv2d(
            hidden_states_207,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_207 = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_16 = hidden_states_208.flatten(2)
        hidden_states_208 = None
        hidden_states_209 = flatten_16.transpose(1, 2)
        flatten_16 = None
        hidden_states_210 = torch._C._nn.gelu(hidden_states_209)
        hidden_states_209 = None
        hidden_states_211 = torch.nn.functional.dropout(
            hidden_states_210, 0.0, False, False
        )
        hidden_states_210 = None
        hidden_states_212 = torch._C._nn.linear(
            hidden_states_211,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_211 = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_213 = torch.nn.functional.dropout(
            hidden_states_212, 0.0, False, False
        )
        hidden_states_212 = None
        layer_output_13 = hidden_states_213 + hidden_states_205
        hidden_states_213 = hidden_states_205 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            layer_output_13,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_1_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            layer_norm_47,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_70 = linear_84.view(1, -1, 5, 64)
        linear_84 = None
        query_layer_14 = view_70.transpose(1, 2)
        view_70 = None
        permute_44 = layer_norm_47.permute(0, 2, 1)
        layer_norm_47 = None
        hidden_states_214 = permute_44.reshape(1, 320, 32, 32)
        permute_44 = None
        hidden_states_215 = torch.conv2d(
            hidden_states_214,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_214 = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_31 = hidden_states_215.reshape(1, 320, -1)
        hidden_states_215 = None
        hidden_states_216 = reshape_31.permute(0, 2, 1)
        reshape_31 = None
        hidden_states_217 = torch.nn.functional.layer_norm(
            hidden_states_216,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_216 = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_85 = torch._C._nn.linear(
            hidden_states_217,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_71 = linear_85.view(1, -1, 5, 64)
        linear_85 = None
        key_layer_14 = view_71.transpose(1, 2)
        view_71 = None
        linear_86 = torch._C._nn.linear(
            hidden_states_217,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_217 = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_72 = linear_86.view(1, -1, 5, 64)
        linear_86 = None
        value_layer_14 = view_72.transpose(1, 2)
        view_72 = None
        transpose_90 = key_layer_14.transpose(-1, -2)
        key_layer_14 = None
        attention_scores_28 = torch.matmul(query_layer_14, transpose_90)
        query_layer_14 = transpose_90 = None
        attention_scores_29 = attention_scores_28 / 8.0
        attention_scores_28 = None
        attention_probs_28 = torch.nn.functional.softmax(attention_scores_29, dim=-1)
        attention_scores_29 = None
        attention_probs_29 = torch.nn.functional.dropout(
            attention_probs_28, 0.0, False, False
        )
        attention_probs_28 = None
        context_layer_42 = torch.matmul(attention_probs_29, value_layer_14)
        attention_probs_29 = value_layer_14 = None
        permute_46 = context_layer_42.permute(0, 2, 1, 3)
        context_layer_42 = None
        context_layer_43 = permute_46.contiguous()
        permute_46 = None
        context_layer_44 = context_layer_43.view((1, 1024, 320))
        context_layer_43 = None
        hidden_states_218 = torch._C._nn.linear(
            context_layer_44,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_44 = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_219 = torch.nn.functional.dropout(
            hidden_states_218, 0.0, False, False
        )
        hidden_states_218 = None
        hidden_states_220 = hidden_states_219 + layer_output_13
        hidden_states_219 = layer_output_13 = None
        layer_norm_49 = torch.nn.functional.layer_norm(
            hidden_states_220,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_221 = torch._C._nn.linear(
            layer_norm_49,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_49 = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_91 = hidden_states_221.transpose(1, 2)
        hidden_states_221 = None
        hidden_states_222 = transpose_91.view(1, 1280, 32, 32)
        transpose_91 = None
        hidden_states_223 = torch.conv2d(
            hidden_states_222,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_222 = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_17 = hidden_states_223.flatten(2)
        hidden_states_223 = None
        hidden_states_224 = flatten_17.transpose(1, 2)
        flatten_17 = None
        hidden_states_225 = torch._C._nn.gelu(hidden_states_224)
        hidden_states_224 = None
        hidden_states_226 = torch.nn.functional.dropout(
            hidden_states_225, 0.0, False, False
        )
        hidden_states_225 = None
        hidden_states_227 = torch._C._nn.linear(
            hidden_states_226,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_226 = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_228 = torch.nn.functional.dropout(
            hidden_states_227, 0.0, False, False
        )
        hidden_states_227 = None
        layer_output_14 = hidden_states_228 + hidden_states_220
        hidden_states_228 = hidden_states_220 = None
        layer_norm_50 = torch.nn.functional.layer_norm(
            layer_output_14,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_1_parameters_bias_ = (None)
        linear_90 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_75 = linear_90.view(1, -1, 5, 64)
        linear_90 = None
        query_layer_15 = view_75.transpose(1, 2)
        view_75 = None
        permute_47 = layer_norm_50.permute(0, 2, 1)
        layer_norm_50 = None
        hidden_states_229 = permute_47.reshape(1, 320, 32, 32)
        permute_47 = None
        hidden_states_230 = torch.conv2d(
            hidden_states_229,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_229 = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_33 = hidden_states_230.reshape(1, 320, -1)
        hidden_states_230 = None
        hidden_states_231 = reshape_33.permute(0, 2, 1)
        reshape_33 = None
        hidden_states_232 = torch.nn.functional.layer_norm(
            hidden_states_231,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_231 = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_91 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_76 = linear_91.view(1, -1, 5, 64)
        linear_91 = None
        key_layer_15 = view_76.transpose(1, 2)
        view_76 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_232 = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_77 = linear_92.view(1, -1, 5, 64)
        linear_92 = None
        value_layer_15 = view_77.transpose(1, 2)
        view_77 = None
        transpose_96 = key_layer_15.transpose(-1, -2)
        key_layer_15 = None
        attention_scores_30 = torch.matmul(query_layer_15, transpose_96)
        query_layer_15 = transpose_96 = None
        attention_scores_31 = attention_scores_30 / 8.0
        attention_scores_30 = None
        attention_probs_30 = torch.nn.functional.softmax(attention_scores_31, dim=-1)
        attention_scores_31 = None
        attention_probs_31 = torch.nn.functional.dropout(
            attention_probs_30, 0.0, False, False
        )
        attention_probs_30 = None
        context_layer_45 = torch.matmul(attention_probs_31, value_layer_15)
        attention_probs_31 = value_layer_15 = None
        permute_49 = context_layer_45.permute(0, 2, 1, 3)
        context_layer_45 = None
        context_layer_46 = permute_49.contiguous()
        permute_49 = None
        context_layer_47 = context_layer_46.view((1, 1024, 320))
        context_layer_46 = None
        hidden_states_233 = torch._C._nn.linear(
            context_layer_47,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_47 = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_234 = torch.nn.functional.dropout(
            hidden_states_233, 0.0, False, False
        )
        hidden_states_233 = None
        hidden_states_235 = hidden_states_234 + layer_output_14
        hidden_states_234 = layer_output_14 = None
        layer_norm_52 = torch.nn.functional.layer_norm(
            hidden_states_235,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_236 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_52 = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_97 = hidden_states_236.transpose(1, 2)
        hidden_states_236 = None
        hidden_states_237 = transpose_97.view(1, 1280, 32, 32)
        transpose_97 = None
        hidden_states_238 = torch.conv2d(
            hidden_states_237,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_237 = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_18 = hidden_states_238.flatten(2)
        hidden_states_238 = None
        hidden_states_239 = flatten_18.transpose(1, 2)
        flatten_18 = None
        hidden_states_240 = torch._C._nn.gelu(hidden_states_239)
        hidden_states_239 = None
        hidden_states_241 = torch.nn.functional.dropout(
            hidden_states_240, 0.0, False, False
        )
        hidden_states_240 = None
        hidden_states_242 = torch._C._nn.linear(
            hidden_states_241,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_241 = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_243 = torch.nn.functional.dropout(
            hidden_states_242, 0.0, False, False
        )
        hidden_states_242 = None
        layer_output_15 = hidden_states_243 + hidden_states_235
        hidden_states_243 = hidden_states_235 = None
        layer_norm_53 = torch.nn.functional.layer_norm(
            layer_output_15,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_1_parameters_bias_ = (None)
        linear_96 = torch._C._nn.linear(
            layer_norm_53,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_80 = linear_96.view(1, -1, 5, 64)
        linear_96 = None
        query_layer_16 = view_80.transpose(1, 2)
        view_80 = None
        permute_50 = layer_norm_53.permute(0, 2, 1)
        layer_norm_53 = None
        hidden_states_244 = permute_50.reshape(1, 320, 32, 32)
        permute_50 = None
        hidden_states_245 = torch.conv2d(
            hidden_states_244,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_244 = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_35 = hidden_states_245.reshape(1, 320, -1)
        hidden_states_245 = None
        hidden_states_246 = reshape_35.permute(0, 2, 1)
        reshape_35 = None
        hidden_states_247 = torch.nn.functional.layer_norm(
            hidden_states_246,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_246 = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_97 = torch._C._nn.linear(
            hidden_states_247,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_81 = linear_97.view(1, -1, 5, 64)
        linear_97 = None
        key_layer_16 = view_81.transpose(1, 2)
        view_81 = None
        linear_98 = torch._C._nn.linear(
            hidden_states_247,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_247 = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_82 = linear_98.view(1, -1, 5, 64)
        linear_98 = None
        value_layer_16 = view_82.transpose(1, 2)
        view_82 = None
        transpose_102 = key_layer_16.transpose(-1, -2)
        key_layer_16 = None
        attention_scores_32 = torch.matmul(query_layer_16, transpose_102)
        query_layer_16 = transpose_102 = None
        attention_scores_33 = attention_scores_32 / 8.0
        attention_scores_32 = None
        attention_probs_32 = torch.nn.functional.softmax(attention_scores_33, dim=-1)
        attention_scores_33 = None
        attention_probs_33 = torch.nn.functional.dropout(
            attention_probs_32, 0.0, False, False
        )
        attention_probs_32 = None
        context_layer_48 = torch.matmul(attention_probs_33, value_layer_16)
        attention_probs_33 = value_layer_16 = None
        permute_52 = context_layer_48.permute(0, 2, 1, 3)
        context_layer_48 = None
        context_layer_49 = permute_52.contiguous()
        permute_52 = None
        context_layer_50 = context_layer_49.view((1, 1024, 320))
        context_layer_49 = None
        hidden_states_248 = torch._C._nn.linear(
            context_layer_50,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_50 = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_249 = torch.nn.functional.dropout(
            hidden_states_248, 0.0, False, False
        )
        hidden_states_248 = None
        hidden_states_250 = hidden_states_249 + layer_output_15
        hidden_states_249 = layer_output_15 = None
        layer_norm_55 = torch.nn.functional.layer_norm(
            hidden_states_250,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_251 = torch._C._nn.linear(
            layer_norm_55,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_55 = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_103 = hidden_states_251.transpose(1, 2)
        hidden_states_251 = None
        hidden_states_252 = transpose_103.view(1, 1280, 32, 32)
        transpose_103 = None
        hidden_states_253 = torch.conv2d(
            hidden_states_252,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_252 = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_19 = hidden_states_253.flatten(2)
        hidden_states_253 = None
        hidden_states_254 = flatten_19.transpose(1, 2)
        flatten_19 = None
        hidden_states_255 = torch._C._nn.gelu(hidden_states_254)
        hidden_states_254 = None
        hidden_states_256 = torch.nn.functional.dropout(
            hidden_states_255, 0.0, False, False
        )
        hidden_states_255 = None
        hidden_states_257 = torch._C._nn.linear(
            hidden_states_256,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_256 = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_258 = torch.nn.functional.dropout(
            hidden_states_257, 0.0, False, False
        )
        hidden_states_257 = None
        layer_output_16 = hidden_states_258 + hidden_states_250
        hidden_states_258 = hidden_states_250 = None
        layer_norm_56 = torch.nn.functional.layer_norm(
            layer_output_16,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_1_parameters_bias_ = (None)
        linear_102 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_85 = linear_102.view(1, -1, 5, 64)
        linear_102 = None
        query_layer_17 = view_85.transpose(1, 2)
        view_85 = None
        permute_53 = layer_norm_56.permute(0, 2, 1)
        layer_norm_56 = None
        hidden_states_259 = permute_53.reshape(1, 320, 32, 32)
        permute_53 = None
        hidden_states_260 = torch.conv2d(
            hidden_states_259,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_259 = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_37 = hidden_states_260.reshape(1, 320, -1)
        hidden_states_260 = None
        hidden_states_261 = reshape_37.permute(0, 2, 1)
        reshape_37 = None
        hidden_states_262 = torch.nn.functional.layer_norm(
            hidden_states_261,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_261 = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_103 = torch._C._nn.linear(
            hidden_states_262,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_86 = linear_103.view(1, -1, 5, 64)
        linear_103 = None
        key_layer_17 = view_86.transpose(1, 2)
        view_86 = None
        linear_104 = torch._C._nn.linear(
            hidden_states_262,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_262 = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_87 = linear_104.view(1, -1, 5, 64)
        linear_104 = None
        value_layer_17 = view_87.transpose(1, 2)
        view_87 = None
        transpose_108 = key_layer_17.transpose(-1, -2)
        key_layer_17 = None
        attention_scores_34 = torch.matmul(query_layer_17, transpose_108)
        query_layer_17 = transpose_108 = None
        attention_scores_35 = attention_scores_34 / 8.0
        attention_scores_34 = None
        attention_probs_34 = torch.nn.functional.softmax(attention_scores_35, dim=-1)
        attention_scores_35 = None
        attention_probs_35 = torch.nn.functional.dropout(
            attention_probs_34, 0.0, False, False
        )
        attention_probs_34 = None
        context_layer_51 = torch.matmul(attention_probs_35, value_layer_17)
        attention_probs_35 = value_layer_17 = None
        permute_55 = context_layer_51.permute(0, 2, 1, 3)
        context_layer_51 = None
        context_layer_52 = permute_55.contiguous()
        permute_55 = None
        context_layer_53 = context_layer_52.view((1, 1024, 320))
        context_layer_52 = None
        hidden_states_263 = torch._C._nn.linear(
            context_layer_53,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_53 = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_264 = torch.nn.functional.dropout(
            hidden_states_263, 0.0, False, False
        )
        hidden_states_263 = None
        hidden_states_265 = hidden_states_264 + layer_output_16
        hidden_states_264 = layer_output_16 = None
        layer_norm_58 = torch.nn.functional.layer_norm(
            hidden_states_265,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_266 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_58 = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_109 = hidden_states_266.transpose(1, 2)
        hidden_states_266 = None
        hidden_states_267 = transpose_109.view(1, 1280, 32, 32)
        transpose_109 = None
        hidden_states_268 = torch.conv2d(
            hidden_states_267,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_267 = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_20 = hidden_states_268.flatten(2)
        hidden_states_268 = None
        hidden_states_269 = flatten_20.transpose(1, 2)
        flatten_20 = None
        hidden_states_270 = torch._C._nn.gelu(hidden_states_269)
        hidden_states_269 = None
        hidden_states_271 = torch.nn.functional.dropout(
            hidden_states_270, 0.0, False, False
        )
        hidden_states_270 = None
        hidden_states_272 = torch._C._nn.linear(
            hidden_states_271,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_271 = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_273 = torch.nn.functional.dropout(
            hidden_states_272, 0.0, False, False
        )
        hidden_states_272 = None
        layer_output_17 = hidden_states_273 + hidden_states_265
        hidden_states_273 = hidden_states_265 = None
        layer_norm_59 = torch.nn.functional.layer_norm(
            layer_output_17,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_1_parameters_bias_ = (None)
        linear_108 = torch._C._nn.linear(
            layer_norm_59,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_90 = linear_108.view(1, -1, 5, 64)
        linear_108 = None
        query_layer_18 = view_90.transpose(1, 2)
        view_90 = None
        permute_56 = layer_norm_59.permute(0, 2, 1)
        layer_norm_59 = None
        hidden_states_274 = permute_56.reshape(1, 320, 32, 32)
        permute_56 = None
        hidden_states_275 = torch.conv2d(
            hidden_states_274,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_274 = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_39 = hidden_states_275.reshape(1, 320, -1)
        hidden_states_275 = None
        hidden_states_276 = reshape_39.permute(0, 2, 1)
        reshape_39 = None
        hidden_states_277 = torch.nn.functional.layer_norm(
            hidden_states_276,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_276 = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_109 = torch._C._nn.linear(
            hidden_states_277,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_91 = linear_109.view(1, -1, 5, 64)
        linear_109 = None
        key_layer_18 = view_91.transpose(1, 2)
        view_91 = None
        linear_110 = torch._C._nn.linear(
            hidden_states_277,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_277 = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_92 = linear_110.view(1, -1, 5, 64)
        linear_110 = None
        value_layer_18 = view_92.transpose(1, 2)
        view_92 = None
        transpose_114 = key_layer_18.transpose(-1, -2)
        key_layer_18 = None
        attention_scores_36 = torch.matmul(query_layer_18, transpose_114)
        query_layer_18 = transpose_114 = None
        attention_scores_37 = attention_scores_36 / 8.0
        attention_scores_36 = None
        attention_probs_36 = torch.nn.functional.softmax(attention_scores_37, dim=-1)
        attention_scores_37 = None
        attention_probs_37 = torch.nn.functional.dropout(
            attention_probs_36, 0.0, False, False
        )
        attention_probs_36 = None
        context_layer_54 = torch.matmul(attention_probs_37, value_layer_18)
        attention_probs_37 = value_layer_18 = None
        permute_58 = context_layer_54.permute(0, 2, 1, 3)
        context_layer_54 = None
        context_layer_55 = permute_58.contiguous()
        permute_58 = None
        context_layer_56 = context_layer_55.view((1, 1024, 320))
        context_layer_55 = None
        hidden_states_278 = torch._C._nn.linear(
            context_layer_56,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_56 = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_279 = torch.nn.functional.dropout(
            hidden_states_278, 0.0, False, False
        )
        hidden_states_278 = None
        hidden_states_280 = hidden_states_279 + layer_output_17
        hidden_states_279 = layer_output_17 = None
        layer_norm_61 = torch.nn.functional.layer_norm(
            hidden_states_280,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_281 = torch._C._nn.linear(
            layer_norm_61,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_61 = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_115 = hidden_states_281.transpose(1, 2)
        hidden_states_281 = None
        hidden_states_282 = transpose_115.view(1, 1280, 32, 32)
        transpose_115 = None
        hidden_states_283 = torch.conv2d(
            hidden_states_282,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_282 = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_21 = hidden_states_283.flatten(2)
        hidden_states_283 = None
        hidden_states_284 = flatten_21.transpose(1, 2)
        flatten_21 = None
        hidden_states_285 = torch._C._nn.gelu(hidden_states_284)
        hidden_states_284 = None
        hidden_states_286 = torch.nn.functional.dropout(
            hidden_states_285, 0.0, False, False
        )
        hidden_states_285 = None
        hidden_states_287 = torch._C._nn.linear(
            hidden_states_286,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_286 = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_288 = torch.nn.functional.dropout(
            hidden_states_287, 0.0, False, False
        )
        hidden_states_287 = None
        layer_output_18 = hidden_states_288 + hidden_states_280
        hidden_states_288 = hidden_states_280 = None
        layer_norm_62 = torch.nn.functional.layer_norm(
            layer_output_18,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_1_parameters_bias_ = (None)
        linear_114 = torch._C._nn.linear(
            layer_norm_62,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_95 = linear_114.view(1, -1, 5, 64)
        linear_114 = None
        query_layer_19 = view_95.transpose(1, 2)
        view_95 = None
        permute_59 = layer_norm_62.permute(0, 2, 1)
        layer_norm_62 = None
        hidden_states_289 = permute_59.reshape(1, 320, 32, 32)
        permute_59 = None
        hidden_states_290 = torch.conv2d(
            hidden_states_289,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_289 = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_41 = hidden_states_290.reshape(1, 320, -1)
        hidden_states_290 = None
        hidden_states_291 = reshape_41.permute(0, 2, 1)
        reshape_41 = None
        hidden_states_292 = torch.nn.functional.layer_norm(
            hidden_states_291,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_291 = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_115 = torch._C._nn.linear(
            hidden_states_292,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_96 = linear_115.view(1, -1, 5, 64)
        linear_115 = None
        key_layer_19 = view_96.transpose(1, 2)
        view_96 = None
        linear_116 = torch._C._nn.linear(
            hidden_states_292,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_292 = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_97 = linear_116.view(1, -1, 5, 64)
        linear_116 = None
        value_layer_19 = view_97.transpose(1, 2)
        view_97 = None
        transpose_120 = key_layer_19.transpose(-1, -2)
        key_layer_19 = None
        attention_scores_38 = torch.matmul(query_layer_19, transpose_120)
        query_layer_19 = transpose_120 = None
        attention_scores_39 = attention_scores_38 / 8.0
        attention_scores_38 = None
        attention_probs_38 = torch.nn.functional.softmax(attention_scores_39, dim=-1)
        attention_scores_39 = None
        attention_probs_39 = torch.nn.functional.dropout(
            attention_probs_38, 0.0, False, False
        )
        attention_probs_38 = None
        context_layer_57 = torch.matmul(attention_probs_39, value_layer_19)
        attention_probs_39 = value_layer_19 = None
        permute_61 = context_layer_57.permute(0, 2, 1, 3)
        context_layer_57 = None
        context_layer_58 = permute_61.contiguous()
        permute_61 = None
        context_layer_59 = context_layer_58.view((1, 1024, 320))
        context_layer_58 = None
        hidden_states_293 = torch._C._nn.linear(
            context_layer_59,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_59 = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_294 = torch.nn.functional.dropout(
            hidden_states_293, 0.0, False, False
        )
        hidden_states_293 = None
        hidden_states_295 = hidden_states_294 + layer_output_18
        hidden_states_294 = layer_output_18 = None
        layer_norm_64 = torch.nn.functional.layer_norm(
            hidden_states_295,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_296 = torch._C._nn.linear(
            layer_norm_64,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_64 = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_121 = hidden_states_296.transpose(1, 2)
        hidden_states_296 = None
        hidden_states_297 = transpose_121.view(1, 1280, 32, 32)
        transpose_121 = None
        hidden_states_298 = torch.conv2d(
            hidden_states_297,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_297 = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_22 = hidden_states_298.flatten(2)
        hidden_states_298 = None
        hidden_states_299 = flatten_22.transpose(1, 2)
        flatten_22 = None
        hidden_states_300 = torch._C._nn.gelu(hidden_states_299)
        hidden_states_299 = None
        hidden_states_301 = torch.nn.functional.dropout(
            hidden_states_300, 0.0, False, False
        )
        hidden_states_300 = None
        hidden_states_302 = torch._C._nn.linear(
            hidden_states_301,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_301 = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_303 = torch.nn.functional.dropout(
            hidden_states_302, 0.0, False, False
        )
        hidden_states_302 = None
        layer_output_19 = hidden_states_303 + hidden_states_295
        hidden_states_303 = hidden_states_295 = None
        layer_norm_65 = torch.nn.functional.layer_norm(
            layer_output_19,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_1_parameters_bias_ = (None)
        linear_120 = torch._C._nn.linear(
            layer_norm_65,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_100 = linear_120.view(1, -1, 5, 64)
        linear_120 = None
        query_layer_20 = view_100.transpose(1, 2)
        view_100 = None
        permute_62 = layer_norm_65.permute(0, 2, 1)
        layer_norm_65 = None
        hidden_states_304 = permute_62.reshape(1, 320, 32, 32)
        permute_62 = None
        hidden_states_305 = torch.conv2d(
            hidden_states_304,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_304 = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_43 = hidden_states_305.reshape(1, 320, -1)
        hidden_states_305 = None
        hidden_states_306 = reshape_43.permute(0, 2, 1)
        reshape_43 = None
        hidden_states_307 = torch.nn.functional.layer_norm(
            hidden_states_306,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_306 = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_121 = torch._C._nn.linear(
            hidden_states_307,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_101 = linear_121.view(1, -1, 5, 64)
        linear_121 = None
        key_layer_20 = view_101.transpose(1, 2)
        view_101 = None
        linear_122 = torch._C._nn.linear(
            hidden_states_307,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_307 = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_102 = linear_122.view(1, -1, 5, 64)
        linear_122 = None
        value_layer_20 = view_102.transpose(1, 2)
        view_102 = None
        transpose_126 = key_layer_20.transpose(-1, -2)
        key_layer_20 = None
        attention_scores_40 = torch.matmul(query_layer_20, transpose_126)
        query_layer_20 = transpose_126 = None
        attention_scores_41 = attention_scores_40 / 8.0
        attention_scores_40 = None
        attention_probs_40 = torch.nn.functional.softmax(attention_scores_41, dim=-1)
        attention_scores_41 = None
        attention_probs_41 = torch.nn.functional.dropout(
            attention_probs_40, 0.0, False, False
        )
        attention_probs_40 = None
        context_layer_60 = torch.matmul(attention_probs_41, value_layer_20)
        attention_probs_41 = value_layer_20 = None
        permute_64 = context_layer_60.permute(0, 2, 1, 3)
        context_layer_60 = None
        context_layer_61 = permute_64.contiguous()
        permute_64 = None
        context_layer_62 = context_layer_61.view((1, 1024, 320))
        context_layer_61 = None
        hidden_states_308 = torch._C._nn.linear(
            context_layer_62,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_62 = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_309 = torch.nn.functional.dropout(
            hidden_states_308, 0.0, False, False
        )
        hidden_states_308 = None
        hidden_states_310 = hidden_states_309 + layer_output_19
        hidden_states_309 = layer_output_19 = None
        layer_norm_67 = torch.nn.functional.layer_norm(
            hidden_states_310,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_311 = torch._C._nn.linear(
            layer_norm_67,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_67 = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_127 = hidden_states_311.transpose(1, 2)
        hidden_states_311 = None
        hidden_states_312 = transpose_127.view(1, 1280, 32, 32)
        transpose_127 = None
        hidden_states_313 = torch.conv2d(
            hidden_states_312,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_312 = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_23 = hidden_states_313.flatten(2)
        hidden_states_313 = None
        hidden_states_314 = flatten_23.transpose(1, 2)
        flatten_23 = None
        hidden_states_315 = torch._C._nn.gelu(hidden_states_314)
        hidden_states_314 = None
        hidden_states_316 = torch.nn.functional.dropout(
            hidden_states_315, 0.0, False, False
        )
        hidden_states_315 = None
        hidden_states_317 = torch._C._nn.linear(
            hidden_states_316,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_316 = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_318 = torch.nn.functional.dropout(
            hidden_states_317, 0.0, False, False
        )
        hidden_states_317 = None
        layer_output_20 = hidden_states_318 + hidden_states_310
        hidden_states_318 = hidden_states_310 = None
        layer_norm_68 = torch.nn.functional.layer_norm(
            layer_output_20,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_1_parameters_bias_ = (None)
        linear_126 = torch._C._nn.linear(
            layer_norm_68,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_105 = linear_126.view(1, -1, 5, 64)
        linear_126 = None
        query_layer_21 = view_105.transpose(1, 2)
        view_105 = None
        permute_65 = layer_norm_68.permute(0, 2, 1)
        layer_norm_68 = None
        hidden_states_319 = permute_65.reshape(1, 320, 32, 32)
        permute_65 = None
        hidden_states_320 = torch.conv2d(
            hidden_states_319,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_319 = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_45 = hidden_states_320.reshape(1, 320, -1)
        hidden_states_320 = None
        hidden_states_321 = reshape_45.permute(0, 2, 1)
        reshape_45 = None
        hidden_states_322 = torch.nn.functional.layer_norm(
            hidden_states_321,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_321 = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_127 = torch._C._nn.linear(
            hidden_states_322,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_106 = linear_127.view(1, -1, 5, 64)
        linear_127 = None
        key_layer_21 = view_106.transpose(1, 2)
        view_106 = None
        linear_128 = torch._C._nn.linear(
            hidden_states_322,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_322 = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_107 = linear_128.view(1, -1, 5, 64)
        linear_128 = None
        value_layer_21 = view_107.transpose(1, 2)
        view_107 = None
        transpose_132 = key_layer_21.transpose(-1, -2)
        key_layer_21 = None
        attention_scores_42 = torch.matmul(query_layer_21, transpose_132)
        query_layer_21 = transpose_132 = None
        attention_scores_43 = attention_scores_42 / 8.0
        attention_scores_42 = None
        attention_probs_42 = torch.nn.functional.softmax(attention_scores_43, dim=-1)
        attention_scores_43 = None
        attention_probs_43 = torch.nn.functional.dropout(
            attention_probs_42, 0.0, False, False
        )
        attention_probs_42 = None
        context_layer_63 = torch.matmul(attention_probs_43, value_layer_21)
        attention_probs_43 = value_layer_21 = None
        permute_67 = context_layer_63.permute(0, 2, 1, 3)
        context_layer_63 = None
        context_layer_64 = permute_67.contiguous()
        permute_67 = None
        context_layer_65 = context_layer_64.view((1, 1024, 320))
        context_layer_64 = None
        hidden_states_323 = torch._C._nn.linear(
            context_layer_65,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_65 = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_324 = torch.nn.functional.dropout(
            hidden_states_323, 0.0, False, False
        )
        hidden_states_323 = None
        hidden_states_325 = hidden_states_324 + layer_output_20
        hidden_states_324 = layer_output_20 = None
        layer_norm_70 = torch.nn.functional.layer_norm(
            hidden_states_325,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_326 = torch._C._nn.linear(
            layer_norm_70,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_70 = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_133 = hidden_states_326.transpose(1, 2)
        hidden_states_326 = None
        hidden_states_327 = transpose_133.view(1, 1280, 32, 32)
        transpose_133 = None
        hidden_states_328 = torch.conv2d(
            hidden_states_327,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_327 = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_24 = hidden_states_328.flatten(2)
        hidden_states_328 = None
        hidden_states_329 = flatten_24.transpose(1, 2)
        flatten_24 = None
        hidden_states_330 = torch._C._nn.gelu(hidden_states_329)
        hidden_states_329 = None
        hidden_states_331 = torch.nn.functional.dropout(
            hidden_states_330, 0.0, False, False
        )
        hidden_states_330 = None
        hidden_states_332 = torch._C._nn.linear(
            hidden_states_331,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_331 = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_333 = torch.nn.functional.dropout(
            hidden_states_332, 0.0, False, False
        )
        hidden_states_332 = None
        layer_output_21 = hidden_states_333 + hidden_states_325
        hidden_states_333 = hidden_states_325 = None
        layer_norm_71 = torch.nn.functional.layer_norm(
            layer_output_21,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_1_parameters_bias_ = (None)
        linear_132 = torch._C._nn.linear(
            layer_norm_71,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_110 = linear_132.view(1, -1, 5, 64)
        linear_132 = None
        query_layer_22 = view_110.transpose(1, 2)
        view_110 = None
        permute_68 = layer_norm_71.permute(0, 2, 1)
        layer_norm_71 = None
        hidden_states_334 = permute_68.reshape(1, 320, 32, 32)
        permute_68 = None
        hidden_states_335 = torch.conv2d(
            hidden_states_334,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_334 = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_47 = hidden_states_335.reshape(1, 320, -1)
        hidden_states_335 = None
        hidden_states_336 = reshape_47.permute(0, 2, 1)
        reshape_47 = None
        hidden_states_337 = torch.nn.functional.layer_norm(
            hidden_states_336,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_336 = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_133 = torch._C._nn.linear(
            hidden_states_337,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_111 = linear_133.view(1, -1, 5, 64)
        linear_133 = None
        key_layer_22 = view_111.transpose(1, 2)
        view_111 = None
        linear_134 = torch._C._nn.linear(
            hidden_states_337,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_337 = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_112 = linear_134.view(1, -1, 5, 64)
        linear_134 = None
        value_layer_22 = view_112.transpose(1, 2)
        view_112 = None
        transpose_138 = key_layer_22.transpose(-1, -2)
        key_layer_22 = None
        attention_scores_44 = torch.matmul(query_layer_22, transpose_138)
        query_layer_22 = transpose_138 = None
        attention_scores_45 = attention_scores_44 / 8.0
        attention_scores_44 = None
        attention_probs_44 = torch.nn.functional.softmax(attention_scores_45, dim=-1)
        attention_scores_45 = None
        attention_probs_45 = torch.nn.functional.dropout(
            attention_probs_44, 0.0, False, False
        )
        attention_probs_44 = None
        context_layer_66 = torch.matmul(attention_probs_45, value_layer_22)
        attention_probs_45 = value_layer_22 = None
        permute_70 = context_layer_66.permute(0, 2, 1, 3)
        context_layer_66 = None
        context_layer_67 = permute_70.contiguous()
        permute_70 = None
        context_layer_68 = context_layer_67.view((1, 1024, 320))
        context_layer_67 = None
        hidden_states_338 = torch._C._nn.linear(
            context_layer_68,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_68 = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_339 = torch.nn.functional.dropout(
            hidden_states_338, 0.0, False, False
        )
        hidden_states_338 = None
        hidden_states_340 = hidden_states_339 + layer_output_21
        hidden_states_339 = layer_output_21 = None
        layer_norm_73 = torch.nn.functional.layer_norm(
            hidden_states_340,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_341 = torch._C._nn.linear(
            layer_norm_73,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_73 = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_139 = hidden_states_341.transpose(1, 2)
        hidden_states_341 = None
        hidden_states_342 = transpose_139.view(1, 1280, 32, 32)
        transpose_139 = None
        hidden_states_343 = torch.conv2d(
            hidden_states_342,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_342 = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_25 = hidden_states_343.flatten(2)
        hidden_states_343 = None
        hidden_states_344 = flatten_25.transpose(1, 2)
        flatten_25 = None
        hidden_states_345 = torch._C._nn.gelu(hidden_states_344)
        hidden_states_344 = None
        hidden_states_346 = torch.nn.functional.dropout(
            hidden_states_345, 0.0, False, False
        )
        hidden_states_345 = None
        hidden_states_347 = torch._C._nn.linear(
            hidden_states_346,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_346 = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_348 = torch.nn.functional.dropout(
            hidden_states_347, 0.0, False, False
        )
        hidden_states_347 = None
        layer_output_22 = hidden_states_348 + hidden_states_340
        hidden_states_348 = hidden_states_340 = None
        layer_norm_74 = torch.nn.functional.layer_norm(
            layer_output_22,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_1_parameters_bias_ = (None)
        linear_138 = torch._C._nn.linear(
            layer_norm_74,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_115 = linear_138.view(1, -1, 5, 64)
        linear_138 = None
        query_layer_23 = view_115.transpose(1, 2)
        view_115 = None
        permute_71 = layer_norm_74.permute(0, 2, 1)
        layer_norm_74 = None
        hidden_states_349 = permute_71.reshape(1, 320, 32, 32)
        permute_71 = None
        hidden_states_350 = torch.conv2d(
            hidden_states_349,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_349 = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_49 = hidden_states_350.reshape(1, 320, -1)
        hidden_states_350 = None
        hidden_states_351 = reshape_49.permute(0, 2, 1)
        reshape_49 = None
        hidden_states_352 = torch.nn.functional.layer_norm(
            hidden_states_351,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_351 = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_139 = torch._C._nn.linear(
            hidden_states_352,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_116 = linear_139.view(1, -1, 5, 64)
        linear_139 = None
        key_layer_23 = view_116.transpose(1, 2)
        view_116 = None
        linear_140 = torch._C._nn.linear(
            hidden_states_352,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_352 = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_117 = linear_140.view(1, -1, 5, 64)
        linear_140 = None
        value_layer_23 = view_117.transpose(1, 2)
        view_117 = None
        transpose_144 = key_layer_23.transpose(-1, -2)
        key_layer_23 = None
        attention_scores_46 = torch.matmul(query_layer_23, transpose_144)
        query_layer_23 = transpose_144 = None
        attention_scores_47 = attention_scores_46 / 8.0
        attention_scores_46 = None
        attention_probs_46 = torch.nn.functional.softmax(attention_scores_47, dim=-1)
        attention_scores_47 = None
        attention_probs_47 = torch.nn.functional.dropout(
            attention_probs_46, 0.0, False, False
        )
        attention_probs_46 = None
        context_layer_69 = torch.matmul(attention_probs_47, value_layer_23)
        attention_probs_47 = value_layer_23 = None
        permute_73 = context_layer_69.permute(0, 2, 1, 3)
        context_layer_69 = None
        context_layer_70 = permute_73.contiguous()
        permute_73 = None
        context_layer_71 = context_layer_70.view((1, 1024, 320))
        context_layer_70 = None
        hidden_states_353 = torch._C._nn.linear(
            context_layer_71,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_71 = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_354 = torch.nn.functional.dropout(
            hidden_states_353, 0.0, False, False
        )
        hidden_states_353 = None
        hidden_states_355 = hidden_states_354 + layer_output_22
        hidden_states_354 = layer_output_22 = None
        layer_norm_76 = torch.nn.functional.layer_norm(
            hidden_states_355,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_356 = torch._C._nn.linear(
            layer_norm_76,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_76 = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_145 = hidden_states_356.transpose(1, 2)
        hidden_states_356 = None
        hidden_states_357 = transpose_145.view(1, 1280, 32, 32)
        transpose_145 = None
        hidden_states_358 = torch.conv2d(
            hidden_states_357,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_357 = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_26 = hidden_states_358.flatten(2)
        hidden_states_358 = None
        hidden_states_359 = flatten_26.transpose(1, 2)
        flatten_26 = None
        hidden_states_360 = torch._C._nn.gelu(hidden_states_359)
        hidden_states_359 = None
        hidden_states_361 = torch.nn.functional.dropout(
            hidden_states_360, 0.0, False, False
        )
        hidden_states_360 = None
        hidden_states_362 = torch._C._nn.linear(
            hidden_states_361,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_361 = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_363 = torch.nn.functional.dropout(
            hidden_states_362, 0.0, False, False
        )
        hidden_states_362 = None
        layer_output_23 = hidden_states_363 + hidden_states_355
        hidden_states_363 = hidden_states_355 = None
        layer_norm_77 = torch.nn.functional.layer_norm(
            layer_output_23,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_1_parameters_bias_ = (None)
        linear_144 = torch._C._nn.linear(
            layer_norm_77,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_120 = linear_144.view(1, -1, 5, 64)
        linear_144 = None
        query_layer_24 = view_120.transpose(1, 2)
        view_120 = None
        permute_74 = layer_norm_77.permute(0, 2, 1)
        layer_norm_77 = None
        hidden_states_364 = permute_74.reshape(1, 320, 32, 32)
        permute_74 = None
        hidden_states_365 = torch.conv2d(
            hidden_states_364,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_364 = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_51 = hidden_states_365.reshape(1, 320, -1)
        hidden_states_365 = None
        hidden_states_366 = reshape_51.permute(0, 2, 1)
        reshape_51 = None
        hidden_states_367 = torch.nn.functional.layer_norm(
            hidden_states_366,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_366 = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_145 = torch._C._nn.linear(
            hidden_states_367,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_121 = linear_145.view(1, -1, 5, 64)
        linear_145 = None
        key_layer_24 = view_121.transpose(1, 2)
        view_121 = None
        linear_146 = torch._C._nn.linear(
            hidden_states_367,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_367 = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_122 = linear_146.view(1, -1, 5, 64)
        linear_146 = None
        value_layer_24 = view_122.transpose(1, 2)
        view_122 = None
        transpose_150 = key_layer_24.transpose(-1, -2)
        key_layer_24 = None
        attention_scores_48 = torch.matmul(query_layer_24, transpose_150)
        query_layer_24 = transpose_150 = None
        attention_scores_49 = attention_scores_48 / 8.0
        attention_scores_48 = None
        attention_probs_48 = torch.nn.functional.softmax(attention_scores_49, dim=-1)
        attention_scores_49 = None
        attention_probs_49 = torch.nn.functional.dropout(
            attention_probs_48, 0.0, False, False
        )
        attention_probs_48 = None
        context_layer_72 = torch.matmul(attention_probs_49, value_layer_24)
        attention_probs_49 = value_layer_24 = None
        permute_76 = context_layer_72.permute(0, 2, 1, 3)
        context_layer_72 = None
        context_layer_73 = permute_76.contiguous()
        permute_76 = None
        context_layer_74 = context_layer_73.view((1, 1024, 320))
        context_layer_73 = None
        hidden_states_368 = torch._C._nn.linear(
            context_layer_74,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_74 = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_369 = torch.nn.functional.dropout(
            hidden_states_368, 0.0, False, False
        )
        hidden_states_368 = None
        hidden_states_370 = hidden_states_369 + layer_output_23
        hidden_states_369 = layer_output_23 = None
        layer_norm_79 = torch.nn.functional.layer_norm(
            hidden_states_370,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_371 = torch._C._nn.linear(
            layer_norm_79,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_79 = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_151 = hidden_states_371.transpose(1, 2)
        hidden_states_371 = None
        hidden_states_372 = transpose_151.view(1, 1280, 32, 32)
        transpose_151 = None
        hidden_states_373 = torch.conv2d(
            hidden_states_372,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_372 = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_27 = hidden_states_373.flatten(2)
        hidden_states_373 = None
        hidden_states_374 = flatten_27.transpose(1, 2)
        flatten_27 = None
        hidden_states_375 = torch._C._nn.gelu(hidden_states_374)
        hidden_states_374 = None
        hidden_states_376 = torch.nn.functional.dropout(
            hidden_states_375, 0.0, False, False
        )
        hidden_states_375 = None
        hidden_states_377 = torch._C._nn.linear(
            hidden_states_376,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_376 = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_378 = torch.nn.functional.dropout(
            hidden_states_377, 0.0, False, False
        )
        hidden_states_377 = None
        layer_output_24 = hidden_states_378 + hidden_states_370
        hidden_states_378 = hidden_states_370 = None
        layer_norm_80 = torch.nn.functional.layer_norm(
            layer_output_24,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_1_parameters_bias_ = (None)
        linear_150 = torch._C._nn.linear(
            layer_norm_80,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_125 = linear_150.view(1, -1, 5, 64)
        linear_150 = None
        query_layer_25 = view_125.transpose(1, 2)
        view_125 = None
        permute_77 = layer_norm_80.permute(0, 2, 1)
        layer_norm_80 = None
        hidden_states_379 = permute_77.reshape(1, 320, 32, 32)
        permute_77 = None
        hidden_states_380 = torch.conv2d(
            hidden_states_379,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_379 = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_53 = hidden_states_380.reshape(1, 320, -1)
        hidden_states_380 = None
        hidden_states_381 = reshape_53.permute(0, 2, 1)
        reshape_53 = None
        hidden_states_382 = torch.nn.functional.layer_norm(
            hidden_states_381,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_381 = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_151 = torch._C._nn.linear(
            hidden_states_382,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_126 = linear_151.view(1, -1, 5, 64)
        linear_151 = None
        key_layer_25 = view_126.transpose(1, 2)
        view_126 = None
        linear_152 = torch._C._nn.linear(
            hidden_states_382,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_382 = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_127 = linear_152.view(1, -1, 5, 64)
        linear_152 = None
        value_layer_25 = view_127.transpose(1, 2)
        view_127 = None
        transpose_156 = key_layer_25.transpose(-1, -2)
        key_layer_25 = None
        attention_scores_50 = torch.matmul(query_layer_25, transpose_156)
        query_layer_25 = transpose_156 = None
        attention_scores_51 = attention_scores_50 / 8.0
        attention_scores_50 = None
        attention_probs_50 = torch.nn.functional.softmax(attention_scores_51, dim=-1)
        attention_scores_51 = None
        attention_probs_51 = torch.nn.functional.dropout(
            attention_probs_50, 0.0, False, False
        )
        attention_probs_50 = None
        context_layer_75 = torch.matmul(attention_probs_51, value_layer_25)
        attention_probs_51 = value_layer_25 = None
        permute_79 = context_layer_75.permute(0, 2, 1, 3)
        context_layer_75 = None
        context_layer_76 = permute_79.contiguous()
        permute_79 = None
        context_layer_77 = context_layer_76.view((1, 1024, 320))
        context_layer_76 = None
        hidden_states_383 = torch._C._nn.linear(
            context_layer_77,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_77 = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_384 = torch.nn.functional.dropout(
            hidden_states_383, 0.0, False, False
        )
        hidden_states_383 = None
        hidden_states_385 = hidden_states_384 + layer_output_24
        hidden_states_384 = layer_output_24 = None
        layer_norm_82 = torch.nn.functional.layer_norm(
            hidden_states_385,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_386 = torch._C._nn.linear(
            layer_norm_82,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_82 = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_157 = hidden_states_386.transpose(1, 2)
        hidden_states_386 = None
        hidden_states_387 = transpose_157.view(1, 1280, 32, 32)
        transpose_157 = None
        hidden_states_388 = torch.conv2d(
            hidden_states_387,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_387 = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_28 = hidden_states_388.flatten(2)
        hidden_states_388 = None
        hidden_states_389 = flatten_28.transpose(1, 2)
        flatten_28 = None
        hidden_states_390 = torch._C._nn.gelu(hidden_states_389)
        hidden_states_389 = None
        hidden_states_391 = torch.nn.functional.dropout(
            hidden_states_390, 0.0, False, False
        )
        hidden_states_390 = None
        hidden_states_392 = torch._C._nn.linear(
            hidden_states_391,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_391 = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_393 = torch.nn.functional.dropout(
            hidden_states_392, 0.0, False, False
        )
        hidden_states_392 = None
        layer_output_25 = hidden_states_393 + hidden_states_385
        hidden_states_393 = hidden_states_385 = None
        layer_norm_83 = torch.nn.functional.layer_norm(
            layer_output_25,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_1_parameters_bias_ = (None)
        linear_156 = torch._C._nn.linear(
            layer_norm_83,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_130 = linear_156.view(1, -1, 5, 64)
        linear_156 = None
        query_layer_26 = view_130.transpose(1, 2)
        view_130 = None
        permute_80 = layer_norm_83.permute(0, 2, 1)
        layer_norm_83 = None
        hidden_states_394 = permute_80.reshape(1, 320, 32, 32)
        permute_80 = None
        hidden_states_395 = torch.conv2d(
            hidden_states_394,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_394 = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_55 = hidden_states_395.reshape(1, 320, -1)
        hidden_states_395 = None
        hidden_states_396 = reshape_55.permute(0, 2, 1)
        reshape_55 = None
        hidden_states_397 = torch.nn.functional.layer_norm(
            hidden_states_396,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_396 = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_157 = torch._C._nn.linear(
            hidden_states_397,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_131 = linear_157.view(1, -1, 5, 64)
        linear_157 = None
        key_layer_26 = view_131.transpose(1, 2)
        view_131 = None
        linear_158 = torch._C._nn.linear(
            hidden_states_397,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_397 = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_132 = linear_158.view(1, -1, 5, 64)
        linear_158 = None
        value_layer_26 = view_132.transpose(1, 2)
        view_132 = None
        transpose_162 = key_layer_26.transpose(-1, -2)
        key_layer_26 = None
        attention_scores_52 = torch.matmul(query_layer_26, transpose_162)
        query_layer_26 = transpose_162 = None
        attention_scores_53 = attention_scores_52 / 8.0
        attention_scores_52 = None
        attention_probs_52 = torch.nn.functional.softmax(attention_scores_53, dim=-1)
        attention_scores_53 = None
        attention_probs_53 = torch.nn.functional.dropout(
            attention_probs_52, 0.0, False, False
        )
        attention_probs_52 = None
        context_layer_78 = torch.matmul(attention_probs_53, value_layer_26)
        attention_probs_53 = value_layer_26 = None
        permute_82 = context_layer_78.permute(0, 2, 1, 3)
        context_layer_78 = None
        context_layer_79 = permute_82.contiguous()
        permute_82 = None
        context_layer_80 = context_layer_79.view((1, 1024, 320))
        context_layer_79 = None
        hidden_states_398 = torch._C._nn.linear(
            context_layer_80,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_80 = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_399 = torch.nn.functional.dropout(
            hidden_states_398, 0.0, False, False
        )
        hidden_states_398 = None
        hidden_states_400 = hidden_states_399 + layer_output_25
        hidden_states_399 = layer_output_25 = None
        layer_norm_85 = torch.nn.functional.layer_norm(
            hidden_states_400,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_401 = torch._C._nn.linear(
            layer_norm_85,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_85 = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_163 = hidden_states_401.transpose(1, 2)
        hidden_states_401 = None
        hidden_states_402 = transpose_163.view(1, 1280, 32, 32)
        transpose_163 = None
        hidden_states_403 = torch.conv2d(
            hidden_states_402,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_402 = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_29 = hidden_states_403.flatten(2)
        hidden_states_403 = None
        hidden_states_404 = flatten_29.transpose(1, 2)
        flatten_29 = None
        hidden_states_405 = torch._C._nn.gelu(hidden_states_404)
        hidden_states_404 = None
        hidden_states_406 = torch.nn.functional.dropout(
            hidden_states_405, 0.0, False, False
        )
        hidden_states_405 = None
        hidden_states_407 = torch._C._nn.linear(
            hidden_states_406,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_406 = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_408 = torch.nn.functional.dropout(
            hidden_states_407, 0.0, False, False
        )
        hidden_states_407 = None
        layer_output_26 = hidden_states_408 + hidden_states_400
        hidden_states_408 = hidden_states_400 = None
        layer_norm_86 = torch.nn.functional.layer_norm(
            layer_output_26,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_1_parameters_bias_ = (None)
        linear_162 = torch._C._nn.linear(
            layer_norm_86,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_135 = linear_162.view(1, -1, 5, 64)
        linear_162 = None
        query_layer_27 = view_135.transpose(1, 2)
        view_135 = None
        permute_83 = layer_norm_86.permute(0, 2, 1)
        layer_norm_86 = None
        hidden_states_409 = permute_83.reshape(1, 320, 32, 32)
        permute_83 = None
        hidden_states_410 = torch.conv2d(
            hidden_states_409,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_409 = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_57 = hidden_states_410.reshape(1, 320, -1)
        hidden_states_410 = None
        hidden_states_411 = reshape_57.permute(0, 2, 1)
        reshape_57 = None
        hidden_states_412 = torch.nn.functional.layer_norm(
            hidden_states_411,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_411 = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_163 = torch._C._nn.linear(
            hidden_states_412,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_136 = linear_163.view(1, -1, 5, 64)
        linear_163 = None
        key_layer_27 = view_136.transpose(1, 2)
        view_136 = None
        linear_164 = torch._C._nn.linear(
            hidden_states_412,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_412 = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_137 = linear_164.view(1, -1, 5, 64)
        linear_164 = None
        value_layer_27 = view_137.transpose(1, 2)
        view_137 = None
        transpose_168 = key_layer_27.transpose(-1, -2)
        key_layer_27 = None
        attention_scores_54 = torch.matmul(query_layer_27, transpose_168)
        query_layer_27 = transpose_168 = None
        attention_scores_55 = attention_scores_54 / 8.0
        attention_scores_54 = None
        attention_probs_54 = torch.nn.functional.softmax(attention_scores_55, dim=-1)
        attention_scores_55 = None
        attention_probs_55 = torch.nn.functional.dropout(
            attention_probs_54, 0.0, False, False
        )
        attention_probs_54 = None
        context_layer_81 = torch.matmul(attention_probs_55, value_layer_27)
        attention_probs_55 = value_layer_27 = None
        permute_85 = context_layer_81.permute(0, 2, 1, 3)
        context_layer_81 = None
        context_layer_82 = permute_85.contiguous()
        permute_85 = None
        context_layer_83 = context_layer_82.view((1, 1024, 320))
        context_layer_82 = None
        hidden_states_413 = torch._C._nn.linear(
            context_layer_83,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_83 = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_414 = torch.nn.functional.dropout(
            hidden_states_413, 0.0, False, False
        )
        hidden_states_413 = None
        hidden_states_415 = hidden_states_414 + layer_output_26
        hidden_states_414 = layer_output_26 = None
        layer_norm_88 = torch.nn.functional.layer_norm(
            hidden_states_415,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_416 = torch._C._nn.linear(
            layer_norm_88,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_88 = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_169 = hidden_states_416.transpose(1, 2)
        hidden_states_416 = None
        hidden_states_417 = transpose_169.view(1, 1280, 32, 32)
        transpose_169 = None
        hidden_states_418 = torch.conv2d(
            hidden_states_417,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_417 = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_30 = hidden_states_418.flatten(2)
        hidden_states_418 = None
        hidden_states_419 = flatten_30.transpose(1, 2)
        flatten_30 = None
        hidden_states_420 = torch._C._nn.gelu(hidden_states_419)
        hidden_states_419 = None
        hidden_states_421 = torch.nn.functional.dropout(
            hidden_states_420, 0.0, False, False
        )
        hidden_states_420 = None
        hidden_states_422 = torch._C._nn.linear(
            hidden_states_421,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_421 = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_423 = torch.nn.functional.dropout(
            hidden_states_422, 0.0, False, False
        )
        hidden_states_422 = None
        layer_output_27 = hidden_states_423 + hidden_states_415
        hidden_states_423 = hidden_states_415 = None
        layer_norm_89 = torch.nn.functional.layer_norm(
            layer_output_27,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_1_parameters_bias_ = (None)
        linear_168 = torch._C._nn.linear(
            layer_norm_89,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_140 = linear_168.view(1, -1, 5, 64)
        linear_168 = None
        query_layer_28 = view_140.transpose(1, 2)
        view_140 = None
        permute_86 = layer_norm_89.permute(0, 2, 1)
        layer_norm_89 = None
        hidden_states_424 = permute_86.reshape(1, 320, 32, 32)
        permute_86 = None
        hidden_states_425 = torch.conv2d(
            hidden_states_424,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_424 = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_59 = hidden_states_425.reshape(1, 320, -1)
        hidden_states_425 = None
        hidden_states_426 = reshape_59.permute(0, 2, 1)
        reshape_59 = None
        hidden_states_427 = torch.nn.functional.layer_norm(
            hidden_states_426,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_426 = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_169 = torch._C._nn.linear(
            hidden_states_427,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_141 = linear_169.view(1, -1, 5, 64)
        linear_169 = None
        key_layer_28 = view_141.transpose(1, 2)
        view_141 = None
        linear_170 = torch._C._nn.linear(
            hidden_states_427,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_427 = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_142 = linear_170.view(1, -1, 5, 64)
        linear_170 = None
        value_layer_28 = view_142.transpose(1, 2)
        view_142 = None
        transpose_174 = key_layer_28.transpose(-1, -2)
        key_layer_28 = None
        attention_scores_56 = torch.matmul(query_layer_28, transpose_174)
        query_layer_28 = transpose_174 = None
        attention_scores_57 = attention_scores_56 / 8.0
        attention_scores_56 = None
        attention_probs_56 = torch.nn.functional.softmax(attention_scores_57, dim=-1)
        attention_scores_57 = None
        attention_probs_57 = torch.nn.functional.dropout(
            attention_probs_56, 0.0, False, False
        )
        attention_probs_56 = None
        context_layer_84 = torch.matmul(attention_probs_57, value_layer_28)
        attention_probs_57 = value_layer_28 = None
        permute_88 = context_layer_84.permute(0, 2, 1, 3)
        context_layer_84 = None
        context_layer_85 = permute_88.contiguous()
        permute_88 = None
        context_layer_86 = context_layer_85.view((1, 1024, 320))
        context_layer_85 = None
        hidden_states_428 = torch._C._nn.linear(
            context_layer_86,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_86 = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_429 = torch.nn.functional.dropout(
            hidden_states_428, 0.0, False, False
        )
        hidden_states_428 = None
        hidden_states_430 = hidden_states_429 + layer_output_27
        hidden_states_429 = layer_output_27 = None
        layer_norm_91 = torch.nn.functional.layer_norm(
            hidden_states_430,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_431 = torch._C._nn.linear(
            layer_norm_91,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_91 = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_175 = hidden_states_431.transpose(1, 2)
        hidden_states_431 = None
        hidden_states_432 = transpose_175.view(1, 1280, 32, 32)
        transpose_175 = None
        hidden_states_433 = torch.conv2d(
            hidden_states_432,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_432 = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_31 = hidden_states_433.flatten(2)
        hidden_states_433 = None
        hidden_states_434 = flatten_31.transpose(1, 2)
        flatten_31 = None
        hidden_states_435 = torch._C._nn.gelu(hidden_states_434)
        hidden_states_434 = None
        hidden_states_436 = torch.nn.functional.dropout(
            hidden_states_435, 0.0, False, False
        )
        hidden_states_435 = None
        hidden_states_437 = torch._C._nn.linear(
            hidden_states_436,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_436 = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_438 = torch.nn.functional.dropout(
            hidden_states_437, 0.0, False, False
        )
        hidden_states_437 = None
        layer_output_28 = hidden_states_438 + hidden_states_430
        hidden_states_438 = hidden_states_430 = None
        layer_norm_92 = torch.nn.functional.layer_norm(
            layer_output_28,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_1_parameters_bias_ = (None)
        linear_174 = torch._C._nn.linear(
            layer_norm_92,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_145 = linear_174.view(1, -1, 5, 64)
        linear_174 = None
        query_layer_29 = view_145.transpose(1, 2)
        view_145 = None
        permute_89 = layer_norm_92.permute(0, 2, 1)
        layer_norm_92 = None
        hidden_states_439 = permute_89.reshape(1, 320, 32, 32)
        permute_89 = None
        hidden_states_440 = torch.conv2d(
            hidden_states_439,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_439 = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_61 = hidden_states_440.reshape(1, 320, -1)
        hidden_states_440 = None
        hidden_states_441 = reshape_61.permute(0, 2, 1)
        reshape_61 = None
        hidden_states_442 = torch.nn.functional.layer_norm(
            hidden_states_441,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_441 = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_175 = torch._C._nn.linear(
            hidden_states_442,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_146 = linear_175.view(1, -1, 5, 64)
        linear_175 = None
        key_layer_29 = view_146.transpose(1, 2)
        view_146 = None
        linear_176 = torch._C._nn.linear(
            hidden_states_442,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_442 = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_147 = linear_176.view(1, -1, 5, 64)
        linear_176 = None
        value_layer_29 = view_147.transpose(1, 2)
        view_147 = None
        transpose_180 = key_layer_29.transpose(-1, -2)
        key_layer_29 = None
        attention_scores_58 = torch.matmul(query_layer_29, transpose_180)
        query_layer_29 = transpose_180 = None
        attention_scores_59 = attention_scores_58 / 8.0
        attention_scores_58 = None
        attention_probs_58 = torch.nn.functional.softmax(attention_scores_59, dim=-1)
        attention_scores_59 = None
        attention_probs_59 = torch.nn.functional.dropout(
            attention_probs_58, 0.0, False, False
        )
        attention_probs_58 = None
        context_layer_87 = torch.matmul(attention_probs_59, value_layer_29)
        attention_probs_59 = value_layer_29 = None
        permute_91 = context_layer_87.permute(0, 2, 1, 3)
        context_layer_87 = None
        context_layer_88 = permute_91.contiguous()
        permute_91 = None
        context_layer_89 = context_layer_88.view((1, 1024, 320))
        context_layer_88 = None
        hidden_states_443 = torch._C._nn.linear(
            context_layer_89,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_89 = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_444 = torch.nn.functional.dropout(
            hidden_states_443, 0.0, False, False
        )
        hidden_states_443 = None
        hidden_states_445 = hidden_states_444 + layer_output_28
        hidden_states_444 = layer_output_28 = None
        layer_norm_94 = torch.nn.functional.layer_norm(
            hidden_states_445,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_446 = torch._C._nn.linear(
            layer_norm_94,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_94 = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_181 = hidden_states_446.transpose(1, 2)
        hidden_states_446 = None
        hidden_states_447 = transpose_181.view(1, 1280, 32, 32)
        transpose_181 = None
        hidden_states_448 = torch.conv2d(
            hidden_states_447,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_447 = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_32 = hidden_states_448.flatten(2)
        hidden_states_448 = None
        hidden_states_449 = flatten_32.transpose(1, 2)
        flatten_32 = None
        hidden_states_450 = torch._C._nn.gelu(hidden_states_449)
        hidden_states_449 = None
        hidden_states_451 = torch.nn.functional.dropout(
            hidden_states_450, 0.0, False, False
        )
        hidden_states_450 = None
        hidden_states_452 = torch._C._nn.linear(
            hidden_states_451,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_451 = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_18_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_453 = torch.nn.functional.dropout(
            hidden_states_452, 0.0, False, False
        )
        hidden_states_452 = None
        layer_output_29 = hidden_states_453 + hidden_states_445
        hidden_states_453 = hidden_states_445 = None
        layer_norm_95 = torch.nn.functional.layer_norm(
            layer_output_29,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_1_parameters_bias_ = (None)
        linear_180 = torch._C._nn.linear(
            layer_norm_95,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_150 = linear_180.view(1, -1, 5, 64)
        linear_180 = None
        query_layer_30 = view_150.transpose(1, 2)
        view_150 = None
        permute_92 = layer_norm_95.permute(0, 2, 1)
        layer_norm_95 = None
        hidden_states_454 = permute_92.reshape(1, 320, 32, 32)
        permute_92 = None
        hidden_states_455 = torch.conv2d(
            hidden_states_454,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_454 = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_63 = hidden_states_455.reshape(1, 320, -1)
        hidden_states_455 = None
        hidden_states_456 = reshape_63.permute(0, 2, 1)
        reshape_63 = None
        hidden_states_457 = torch.nn.functional.layer_norm(
            hidden_states_456,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_456 = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_181 = torch._C._nn.linear(
            hidden_states_457,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_151 = linear_181.view(1, -1, 5, 64)
        linear_181 = None
        key_layer_30 = view_151.transpose(1, 2)
        view_151 = None
        linear_182 = torch._C._nn.linear(
            hidden_states_457,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_457 = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_152 = linear_182.view(1, -1, 5, 64)
        linear_182 = None
        value_layer_30 = view_152.transpose(1, 2)
        view_152 = None
        transpose_186 = key_layer_30.transpose(-1, -2)
        key_layer_30 = None
        attention_scores_60 = torch.matmul(query_layer_30, transpose_186)
        query_layer_30 = transpose_186 = None
        attention_scores_61 = attention_scores_60 / 8.0
        attention_scores_60 = None
        attention_probs_60 = torch.nn.functional.softmax(attention_scores_61, dim=-1)
        attention_scores_61 = None
        attention_probs_61 = torch.nn.functional.dropout(
            attention_probs_60, 0.0, False, False
        )
        attention_probs_60 = None
        context_layer_90 = torch.matmul(attention_probs_61, value_layer_30)
        attention_probs_61 = value_layer_30 = None
        permute_94 = context_layer_90.permute(0, 2, 1, 3)
        context_layer_90 = None
        context_layer_91 = permute_94.contiguous()
        permute_94 = None
        context_layer_92 = context_layer_91.view((1, 1024, 320))
        context_layer_91 = None
        hidden_states_458 = torch._C._nn.linear(
            context_layer_92,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_92 = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_459 = torch.nn.functional.dropout(
            hidden_states_458, 0.0, False, False
        )
        hidden_states_458 = None
        hidden_states_460 = hidden_states_459 + layer_output_29
        hidden_states_459 = layer_output_29 = None
        layer_norm_97 = torch.nn.functional.layer_norm(
            hidden_states_460,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_461 = torch._C._nn.linear(
            layer_norm_97,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_97 = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_187 = hidden_states_461.transpose(1, 2)
        hidden_states_461 = None
        hidden_states_462 = transpose_187.view(1, 1280, 32, 32)
        transpose_187 = None
        hidden_states_463 = torch.conv2d(
            hidden_states_462,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_462 = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_33 = hidden_states_463.flatten(2)
        hidden_states_463 = None
        hidden_states_464 = flatten_33.transpose(1, 2)
        flatten_33 = None
        hidden_states_465 = torch._C._nn.gelu(hidden_states_464)
        hidden_states_464 = None
        hidden_states_466 = torch.nn.functional.dropout(
            hidden_states_465, 0.0, False, False
        )
        hidden_states_465 = None
        hidden_states_467 = torch._C._nn.linear(
            hidden_states_466,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_466 = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_19_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_468 = torch.nn.functional.dropout(
            hidden_states_467, 0.0, False, False
        )
        hidden_states_467 = None
        layer_output_30 = hidden_states_468 + hidden_states_460
        hidden_states_468 = hidden_states_460 = None
        layer_norm_98 = torch.nn.functional.layer_norm(
            layer_output_30,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_1_parameters_bias_ = (None)
        linear_186 = torch._C._nn.linear(
            layer_norm_98,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_155 = linear_186.view(1, -1, 5, 64)
        linear_186 = None
        query_layer_31 = view_155.transpose(1, 2)
        view_155 = None
        permute_95 = layer_norm_98.permute(0, 2, 1)
        layer_norm_98 = None
        hidden_states_469 = permute_95.reshape(1, 320, 32, 32)
        permute_95 = None
        hidden_states_470 = torch.conv2d(
            hidden_states_469,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_469 = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_65 = hidden_states_470.reshape(1, 320, -1)
        hidden_states_470 = None
        hidden_states_471 = reshape_65.permute(0, 2, 1)
        reshape_65 = None
        hidden_states_472 = torch.nn.functional.layer_norm(
            hidden_states_471,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_471 = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_187 = torch._C._nn.linear(
            hidden_states_472,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_156 = linear_187.view(1, -1, 5, 64)
        linear_187 = None
        key_layer_31 = view_156.transpose(1, 2)
        view_156 = None
        linear_188 = torch._C._nn.linear(
            hidden_states_472,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_472 = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_157 = linear_188.view(1, -1, 5, 64)
        linear_188 = None
        value_layer_31 = view_157.transpose(1, 2)
        view_157 = None
        transpose_192 = key_layer_31.transpose(-1, -2)
        key_layer_31 = None
        attention_scores_62 = torch.matmul(query_layer_31, transpose_192)
        query_layer_31 = transpose_192 = None
        attention_scores_63 = attention_scores_62 / 8.0
        attention_scores_62 = None
        attention_probs_62 = torch.nn.functional.softmax(attention_scores_63, dim=-1)
        attention_scores_63 = None
        attention_probs_63 = torch.nn.functional.dropout(
            attention_probs_62, 0.0, False, False
        )
        attention_probs_62 = None
        context_layer_93 = torch.matmul(attention_probs_63, value_layer_31)
        attention_probs_63 = value_layer_31 = None
        permute_97 = context_layer_93.permute(0, 2, 1, 3)
        context_layer_93 = None
        context_layer_94 = permute_97.contiguous()
        permute_97 = None
        context_layer_95 = context_layer_94.view((1, 1024, 320))
        context_layer_94 = None
        hidden_states_473 = torch._C._nn.linear(
            context_layer_95,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_95 = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_474 = torch.nn.functional.dropout(
            hidden_states_473, 0.0, False, False
        )
        hidden_states_473 = None
        hidden_states_475 = hidden_states_474 + layer_output_30
        hidden_states_474 = layer_output_30 = None
        layer_norm_100 = torch.nn.functional.layer_norm(
            hidden_states_475,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_476 = torch._C._nn.linear(
            layer_norm_100,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_100 = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_193 = hidden_states_476.transpose(1, 2)
        hidden_states_476 = None
        hidden_states_477 = transpose_193.view(1, 1280, 32, 32)
        transpose_193 = None
        hidden_states_478 = torch.conv2d(
            hidden_states_477,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_477 = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_34 = hidden_states_478.flatten(2)
        hidden_states_478 = None
        hidden_states_479 = flatten_34.transpose(1, 2)
        flatten_34 = None
        hidden_states_480 = torch._C._nn.gelu(hidden_states_479)
        hidden_states_479 = None
        hidden_states_481 = torch.nn.functional.dropout(
            hidden_states_480, 0.0, False, False
        )
        hidden_states_480 = None
        hidden_states_482 = torch._C._nn.linear(
            hidden_states_481,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_481 = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_20_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_483 = torch.nn.functional.dropout(
            hidden_states_482, 0.0, False, False
        )
        hidden_states_482 = None
        layer_output_31 = hidden_states_483 + hidden_states_475
        hidden_states_483 = hidden_states_475 = None
        layer_norm_101 = torch.nn.functional.layer_norm(
            layer_output_31,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_1_parameters_bias_ = (None)
        linear_192 = torch._C._nn.linear(
            layer_norm_101,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_160 = linear_192.view(1, -1, 5, 64)
        linear_192 = None
        query_layer_32 = view_160.transpose(1, 2)
        view_160 = None
        permute_98 = layer_norm_101.permute(0, 2, 1)
        layer_norm_101 = None
        hidden_states_484 = permute_98.reshape(1, 320, 32, 32)
        permute_98 = None
        hidden_states_485 = torch.conv2d(
            hidden_states_484,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_484 = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_67 = hidden_states_485.reshape(1, 320, -1)
        hidden_states_485 = None
        hidden_states_486 = reshape_67.permute(0, 2, 1)
        reshape_67 = None
        hidden_states_487 = torch.nn.functional.layer_norm(
            hidden_states_486,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_486 = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_193 = torch._C._nn.linear(
            hidden_states_487,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_161 = linear_193.view(1, -1, 5, 64)
        linear_193 = None
        key_layer_32 = view_161.transpose(1, 2)
        view_161 = None
        linear_194 = torch._C._nn.linear(
            hidden_states_487,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_487 = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_162 = linear_194.view(1, -1, 5, 64)
        linear_194 = None
        value_layer_32 = view_162.transpose(1, 2)
        view_162 = None
        transpose_198 = key_layer_32.transpose(-1, -2)
        key_layer_32 = None
        attention_scores_64 = torch.matmul(query_layer_32, transpose_198)
        query_layer_32 = transpose_198 = None
        attention_scores_65 = attention_scores_64 / 8.0
        attention_scores_64 = None
        attention_probs_64 = torch.nn.functional.softmax(attention_scores_65, dim=-1)
        attention_scores_65 = None
        attention_probs_65 = torch.nn.functional.dropout(
            attention_probs_64, 0.0, False, False
        )
        attention_probs_64 = None
        context_layer_96 = torch.matmul(attention_probs_65, value_layer_32)
        attention_probs_65 = value_layer_32 = None
        permute_100 = context_layer_96.permute(0, 2, 1, 3)
        context_layer_96 = None
        context_layer_97 = permute_100.contiguous()
        permute_100 = None
        context_layer_98 = context_layer_97.view((1, 1024, 320))
        context_layer_97 = None
        hidden_states_488 = torch._C._nn.linear(
            context_layer_98,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_98 = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_489 = torch.nn.functional.dropout(
            hidden_states_488, 0.0, False, False
        )
        hidden_states_488 = None
        hidden_states_490 = hidden_states_489 + layer_output_31
        hidden_states_489 = layer_output_31 = None
        layer_norm_103 = torch.nn.functional.layer_norm(
            hidden_states_490,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_491 = torch._C._nn.linear(
            layer_norm_103,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_103 = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_199 = hidden_states_491.transpose(1, 2)
        hidden_states_491 = None
        hidden_states_492 = transpose_199.view(1, 1280, 32, 32)
        transpose_199 = None
        hidden_states_493 = torch.conv2d(
            hidden_states_492,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_492 = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_35 = hidden_states_493.flatten(2)
        hidden_states_493 = None
        hidden_states_494 = flatten_35.transpose(1, 2)
        flatten_35 = None
        hidden_states_495 = torch._C._nn.gelu(hidden_states_494)
        hidden_states_494 = None
        hidden_states_496 = torch.nn.functional.dropout(
            hidden_states_495, 0.0, False, False
        )
        hidden_states_495 = None
        hidden_states_497 = torch._C._nn.linear(
            hidden_states_496,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_496 = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_21_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_498 = torch.nn.functional.dropout(
            hidden_states_497, 0.0, False, False
        )
        hidden_states_497 = None
        layer_output_32 = hidden_states_498 + hidden_states_490
        hidden_states_498 = hidden_states_490 = None
        layer_norm_104 = torch.nn.functional.layer_norm(
            layer_output_32,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_1_parameters_bias_ = (None)
        linear_198 = torch._C._nn.linear(
            layer_norm_104,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_165 = linear_198.view(1, -1, 5, 64)
        linear_198 = None
        query_layer_33 = view_165.transpose(1, 2)
        view_165 = None
        permute_101 = layer_norm_104.permute(0, 2, 1)
        layer_norm_104 = None
        hidden_states_499 = permute_101.reshape(1, 320, 32, 32)
        permute_101 = None
        hidden_states_500 = torch.conv2d(
            hidden_states_499,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_499 = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_69 = hidden_states_500.reshape(1, 320, -1)
        hidden_states_500 = None
        hidden_states_501 = reshape_69.permute(0, 2, 1)
        reshape_69 = None
        hidden_states_502 = torch.nn.functional.layer_norm(
            hidden_states_501,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_501 = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_199 = torch._C._nn.linear(
            hidden_states_502,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_166 = linear_199.view(1, -1, 5, 64)
        linear_199 = None
        key_layer_33 = view_166.transpose(1, 2)
        view_166 = None
        linear_200 = torch._C._nn.linear(
            hidden_states_502,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_502 = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_167 = linear_200.view(1, -1, 5, 64)
        linear_200 = None
        value_layer_33 = view_167.transpose(1, 2)
        view_167 = None
        transpose_204 = key_layer_33.transpose(-1, -2)
        key_layer_33 = None
        attention_scores_66 = torch.matmul(query_layer_33, transpose_204)
        query_layer_33 = transpose_204 = None
        attention_scores_67 = attention_scores_66 / 8.0
        attention_scores_66 = None
        attention_probs_66 = torch.nn.functional.softmax(attention_scores_67, dim=-1)
        attention_scores_67 = None
        attention_probs_67 = torch.nn.functional.dropout(
            attention_probs_66, 0.0, False, False
        )
        attention_probs_66 = None
        context_layer_99 = torch.matmul(attention_probs_67, value_layer_33)
        attention_probs_67 = value_layer_33 = None
        permute_103 = context_layer_99.permute(0, 2, 1, 3)
        context_layer_99 = None
        context_layer_100 = permute_103.contiguous()
        permute_103 = None
        context_layer_101 = context_layer_100.view((1, 1024, 320))
        context_layer_100 = None
        hidden_states_503 = torch._C._nn.linear(
            context_layer_101,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_101 = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_504 = torch.nn.functional.dropout(
            hidden_states_503, 0.0, False, False
        )
        hidden_states_503 = None
        hidden_states_505 = hidden_states_504 + layer_output_32
        hidden_states_504 = layer_output_32 = None
        layer_norm_106 = torch.nn.functional.layer_norm(
            hidden_states_505,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_506 = torch._C._nn.linear(
            layer_norm_106,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_106 = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_205 = hidden_states_506.transpose(1, 2)
        hidden_states_506 = None
        hidden_states_507 = transpose_205.view(1, 1280, 32, 32)
        transpose_205 = None
        hidden_states_508 = torch.conv2d(
            hidden_states_507,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_507 = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_36 = hidden_states_508.flatten(2)
        hidden_states_508 = None
        hidden_states_509 = flatten_36.transpose(1, 2)
        flatten_36 = None
        hidden_states_510 = torch._C._nn.gelu(hidden_states_509)
        hidden_states_509 = None
        hidden_states_511 = torch.nn.functional.dropout(
            hidden_states_510, 0.0, False, False
        )
        hidden_states_510 = None
        hidden_states_512 = torch._C._nn.linear(
            hidden_states_511,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_511 = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_22_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_513 = torch.nn.functional.dropout(
            hidden_states_512, 0.0, False, False
        )
        hidden_states_512 = None
        layer_output_33 = hidden_states_513 + hidden_states_505
        hidden_states_513 = hidden_states_505 = None
        layer_norm_107 = torch.nn.functional.layer_norm(
            layer_output_33,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_1_parameters_bias_ = (None)
        linear_204 = torch._C._nn.linear(
            layer_norm_107,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_170 = linear_204.view(1, -1, 5, 64)
        linear_204 = None
        query_layer_34 = view_170.transpose(1, 2)
        view_170 = None
        permute_104 = layer_norm_107.permute(0, 2, 1)
        layer_norm_107 = None
        hidden_states_514 = permute_104.reshape(1, 320, 32, 32)
        permute_104 = None
        hidden_states_515 = torch.conv2d(
            hidden_states_514,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_514 = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_71 = hidden_states_515.reshape(1, 320, -1)
        hidden_states_515 = None
        hidden_states_516 = reshape_71.permute(0, 2, 1)
        reshape_71 = None
        hidden_states_517 = torch.nn.functional.layer_norm(
            hidden_states_516,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_516 = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_205 = torch._C._nn.linear(
            hidden_states_517,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_171 = linear_205.view(1, -1, 5, 64)
        linear_205 = None
        key_layer_34 = view_171.transpose(1, 2)
        view_171 = None
        linear_206 = torch._C._nn.linear(
            hidden_states_517,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_517 = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_172 = linear_206.view(1, -1, 5, 64)
        linear_206 = None
        value_layer_34 = view_172.transpose(1, 2)
        view_172 = None
        transpose_210 = key_layer_34.transpose(-1, -2)
        key_layer_34 = None
        attention_scores_68 = torch.matmul(query_layer_34, transpose_210)
        query_layer_34 = transpose_210 = None
        attention_scores_69 = attention_scores_68 / 8.0
        attention_scores_68 = None
        attention_probs_68 = torch.nn.functional.softmax(attention_scores_69, dim=-1)
        attention_scores_69 = None
        attention_probs_69 = torch.nn.functional.dropout(
            attention_probs_68, 0.0, False, False
        )
        attention_probs_68 = None
        context_layer_102 = torch.matmul(attention_probs_69, value_layer_34)
        attention_probs_69 = value_layer_34 = None
        permute_106 = context_layer_102.permute(0, 2, 1, 3)
        context_layer_102 = None
        context_layer_103 = permute_106.contiguous()
        permute_106 = None
        context_layer_104 = context_layer_103.view((1, 1024, 320))
        context_layer_103 = None
        hidden_states_518 = torch._C._nn.linear(
            context_layer_104,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_104 = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_519 = torch.nn.functional.dropout(
            hidden_states_518, 0.0, False, False
        )
        hidden_states_518 = None
        hidden_states_520 = hidden_states_519 + layer_output_33
        hidden_states_519 = layer_output_33 = None
        layer_norm_109 = torch.nn.functional.layer_norm(
            hidden_states_520,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_521 = torch._C._nn.linear(
            layer_norm_109,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_109 = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_211 = hidden_states_521.transpose(1, 2)
        hidden_states_521 = None
        hidden_states_522 = transpose_211.view(1, 1280, 32, 32)
        transpose_211 = None
        hidden_states_523 = torch.conv2d(
            hidden_states_522,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_522 = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_37 = hidden_states_523.flatten(2)
        hidden_states_523 = None
        hidden_states_524 = flatten_37.transpose(1, 2)
        flatten_37 = None
        hidden_states_525 = torch._C._nn.gelu(hidden_states_524)
        hidden_states_524 = None
        hidden_states_526 = torch.nn.functional.dropout(
            hidden_states_525, 0.0, False, False
        )
        hidden_states_525 = None
        hidden_states_527 = torch._C._nn.linear(
            hidden_states_526,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_526 = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_23_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_528 = torch.nn.functional.dropout(
            hidden_states_527, 0.0, False, False
        )
        hidden_states_527 = None
        layer_output_34 = hidden_states_528 + hidden_states_520
        hidden_states_528 = hidden_states_520 = None
        layer_norm_110 = torch.nn.functional.layer_norm(
            layer_output_34,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_1_parameters_bias_ = (None)
        linear_210 = torch._C._nn.linear(
            layer_norm_110,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_175 = linear_210.view(1, -1, 5, 64)
        linear_210 = None
        query_layer_35 = view_175.transpose(1, 2)
        view_175 = None
        permute_107 = layer_norm_110.permute(0, 2, 1)
        layer_norm_110 = None
        hidden_states_529 = permute_107.reshape(1, 320, 32, 32)
        permute_107 = None
        hidden_states_530 = torch.conv2d(
            hidden_states_529,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_529 = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_73 = hidden_states_530.reshape(1, 320, -1)
        hidden_states_530 = None
        hidden_states_531 = reshape_73.permute(0, 2, 1)
        reshape_73 = None
        hidden_states_532 = torch.nn.functional.layer_norm(
            hidden_states_531,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_531 = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_211 = torch._C._nn.linear(
            hidden_states_532,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_176 = linear_211.view(1, -1, 5, 64)
        linear_211 = None
        key_layer_35 = view_176.transpose(1, 2)
        view_176 = None
        linear_212 = torch._C._nn.linear(
            hidden_states_532,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_532 = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_177 = linear_212.view(1, -1, 5, 64)
        linear_212 = None
        value_layer_35 = view_177.transpose(1, 2)
        view_177 = None
        transpose_216 = key_layer_35.transpose(-1, -2)
        key_layer_35 = None
        attention_scores_70 = torch.matmul(query_layer_35, transpose_216)
        query_layer_35 = transpose_216 = None
        attention_scores_71 = attention_scores_70 / 8.0
        attention_scores_70 = None
        attention_probs_70 = torch.nn.functional.softmax(attention_scores_71, dim=-1)
        attention_scores_71 = None
        attention_probs_71 = torch.nn.functional.dropout(
            attention_probs_70, 0.0, False, False
        )
        attention_probs_70 = None
        context_layer_105 = torch.matmul(attention_probs_71, value_layer_35)
        attention_probs_71 = value_layer_35 = None
        permute_109 = context_layer_105.permute(0, 2, 1, 3)
        context_layer_105 = None
        context_layer_106 = permute_109.contiguous()
        permute_109 = None
        context_layer_107 = context_layer_106.view((1, 1024, 320))
        context_layer_106 = None
        hidden_states_533 = torch._C._nn.linear(
            context_layer_107,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_107 = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_534 = torch.nn.functional.dropout(
            hidden_states_533, 0.0, False, False
        )
        hidden_states_533 = None
        hidden_states_535 = hidden_states_534 + layer_output_34
        hidden_states_534 = layer_output_34 = None
        layer_norm_112 = torch.nn.functional.layer_norm(
            hidden_states_535,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_536 = torch._C._nn.linear(
            layer_norm_112,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_112 = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_217 = hidden_states_536.transpose(1, 2)
        hidden_states_536 = None
        hidden_states_537 = transpose_217.view(1, 1280, 32, 32)
        transpose_217 = None
        hidden_states_538 = torch.conv2d(
            hidden_states_537,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_537 = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_38 = hidden_states_538.flatten(2)
        hidden_states_538 = None
        hidden_states_539 = flatten_38.transpose(1, 2)
        flatten_38 = None
        hidden_states_540 = torch._C._nn.gelu(hidden_states_539)
        hidden_states_539 = None
        hidden_states_541 = torch.nn.functional.dropout(
            hidden_states_540, 0.0, False, False
        )
        hidden_states_540 = None
        hidden_states_542 = torch._C._nn.linear(
            hidden_states_541,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_541 = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_24_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_543 = torch.nn.functional.dropout(
            hidden_states_542, 0.0, False, False
        )
        hidden_states_542 = None
        layer_output_35 = hidden_states_543 + hidden_states_535
        hidden_states_543 = hidden_states_535 = None
        layer_norm_113 = torch.nn.functional.layer_norm(
            layer_output_35,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_1_parameters_bias_ = (None)
        linear_216 = torch._C._nn.linear(
            layer_norm_113,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_180 = linear_216.view(1, -1, 5, 64)
        linear_216 = None
        query_layer_36 = view_180.transpose(1, 2)
        view_180 = None
        permute_110 = layer_norm_113.permute(0, 2, 1)
        layer_norm_113 = None
        hidden_states_544 = permute_110.reshape(1, 320, 32, 32)
        permute_110 = None
        hidden_states_545 = torch.conv2d(
            hidden_states_544,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_544 = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_75 = hidden_states_545.reshape(1, 320, -1)
        hidden_states_545 = None
        hidden_states_546 = reshape_75.permute(0, 2, 1)
        reshape_75 = None
        hidden_states_547 = torch.nn.functional.layer_norm(
            hidden_states_546,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_546 = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_217 = torch._C._nn.linear(
            hidden_states_547,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_181 = linear_217.view(1, -1, 5, 64)
        linear_217 = None
        key_layer_36 = view_181.transpose(1, 2)
        view_181 = None
        linear_218 = torch._C._nn.linear(
            hidden_states_547,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_547 = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_182 = linear_218.view(1, -1, 5, 64)
        linear_218 = None
        value_layer_36 = view_182.transpose(1, 2)
        view_182 = None
        transpose_222 = key_layer_36.transpose(-1, -2)
        key_layer_36 = None
        attention_scores_72 = torch.matmul(query_layer_36, transpose_222)
        query_layer_36 = transpose_222 = None
        attention_scores_73 = attention_scores_72 / 8.0
        attention_scores_72 = None
        attention_probs_72 = torch.nn.functional.softmax(attention_scores_73, dim=-1)
        attention_scores_73 = None
        attention_probs_73 = torch.nn.functional.dropout(
            attention_probs_72, 0.0, False, False
        )
        attention_probs_72 = None
        context_layer_108 = torch.matmul(attention_probs_73, value_layer_36)
        attention_probs_73 = value_layer_36 = None
        permute_112 = context_layer_108.permute(0, 2, 1, 3)
        context_layer_108 = None
        context_layer_109 = permute_112.contiguous()
        permute_112 = None
        context_layer_110 = context_layer_109.view((1, 1024, 320))
        context_layer_109 = None
        hidden_states_548 = torch._C._nn.linear(
            context_layer_110,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_110 = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_549 = torch.nn.functional.dropout(
            hidden_states_548, 0.0, False, False
        )
        hidden_states_548 = None
        hidden_states_550 = hidden_states_549 + layer_output_35
        hidden_states_549 = layer_output_35 = None
        layer_norm_115 = torch.nn.functional.layer_norm(
            hidden_states_550,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_551 = torch._C._nn.linear(
            layer_norm_115,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_115 = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_223 = hidden_states_551.transpose(1, 2)
        hidden_states_551 = None
        hidden_states_552 = transpose_223.view(1, 1280, 32, 32)
        transpose_223 = None
        hidden_states_553 = torch.conv2d(
            hidden_states_552,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_552 = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_39 = hidden_states_553.flatten(2)
        hidden_states_553 = None
        hidden_states_554 = flatten_39.transpose(1, 2)
        flatten_39 = None
        hidden_states_555 = torch._C._nn.gelu(hidden_states_554)
        hidden_states_554 = None
        hidden_states_556 = torch.nn.functional.dropout(
            hidden_states_555, 0.0, False, False
        )
        hidden_states_555 = None
        hidden_states_557 = torch._C._nn.linear(
            hidden_states_556,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_556 = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_25_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_558 = torch.nn.functional.dropout(
            hidden_states_557, 0.0, False, False
        )
        hidden_states_557 = None
        layer_output_36 = hidden_states_558 + hidden_states_550
        hidden_states_558 = hidden_states_550 = None
        layer_norm_116 = torch.nn.functional.layer_norm(
            layer_output_36,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_1_parameters_bias_ = (None)
        linear_222 = torch._C._nn.linear(
            layer_norm_116,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_185 = linear_222.view(1, -1, 5, 64)
        linear_222 = None
        query_layer_37 = view_185.transpose(1, 2)
        view_185 = None
        permute_113 = layer_norm_116.permute(0, 2, 1)
        layer_norm_116 = None
        hidden_states_559 = permute_113.reshape(1, 320, 32, 32)
        permute_113 = None
        hidden_states_560 = torch.conv2d(
            hidden_states_559,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_559 = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_77 = hidden_states_560.reshape(1, 320, -1)
        hidden_states_560 = None
        hidden_states_561 = reshape_77.permute(0, 2, 1)
        reshape_77 = None
        hidden_states_562 = torch.nn.functional.layer_norm(
            hidden_states_561,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_561 = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_223 = torch._C._nn.linear(
            hidden_states_562,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_186 = linear_223.view(1, -1, 5, 64)
        linear_223 = None
        key_layer_37 = view_186.transpose(1, 2)
        view_186 = None
        linear_224 = torch._C._nn.linear(
            hidden_states_562,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_562 = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_187 = linear_224.view(1, -1, 5, 64)
        linear_224 = None
        value_layer_37 = view_187.transpose(1, 2)
        view_187 = None
        transpose_228 = key_layer_37.transpose(-1, -2)
        key_layer_37 = None
        attention_scores_74 = torch.matmul(query_layer_37, transpose_228)
        query_layer_37 = transpose_228 = None
        attention_scores_75 = attention_scores_74 / 8.0
        attention_scores_74 = None
        attention_probs_74 = torch.nn.functional.softmax(attention_scores_75, dim=-1)
        attention_scores_75 = None
        attention_probs_75 = torch.nn.functional.dropout(
            attention_probs_74, 0.0, False, False
        )
        attention_probs_74 = None
        context_layer_111 = torch.matmul(attention_probs_75, value_layer_37)
        attention_probs_75 = value_layer_37 = None
        permute_115 = context_layer_111.permute(0, 2, 1, 3)
        context_layer_111 = None
        context_layer_112 = permute_115.contiguous()
        permute_115 = None
        context_layer_113 = context_layer_112.view((1, 1024, 320))
        context_layer_112 = None
        hidden_states_563 = torch._C._nn.linear(
            context_layer_113,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_113 = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_564 = torch.nn.functional.dropout(
            hidden_states_563, 0.0, False, False
        )
        hidden_states_563 = None
        hidden_states_565 = hidden_states_564 + layer_output_36
        hidden_states_564 = layer_output_36 = None
        layer_norm_118 = torch.nn.functional.layer_norm(
            hidden_states_565,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_566 = torch._C._nn.linear(
            layer_norm_118,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_118 = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_229 = hidden_states_566.transpose(1, 2)
        hidden_states_566 = None
        hidden_states_567 = transpose_229.view(1, 1280, 32, 32)
        transpose_229 = None
        hidden_states_568 = torch.conv2d(
            hidden_states_567,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_567 = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_40 = hidden_states_568.flatten(2)
        hidden_states_568 = None
        hidden_states_569 = flatten_40.transpose(1, 2)
        flatten_40 = None
        hidden_states_570 = torch._C._nn.gelu(hidden_states_569)
        hidden_states_569 = None
        hidden_states_571 = torch.nn.functional.dropout(
            hidden_states_570, 0.0, False, False
        )
        hidden_states_570 = None
        hidden_states_572 = torch._C._nn.linear(
            hidden_states_571,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_571 = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_26_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_573 = torch.nn.functional.dropout(
            hidden_states_572, 0.0, False, False
        )
        hidden_states_572 = None
        layer_output_37 = hidden_states_573 + hidden_states_565
        hidden_states_573 = hidden_states_565 = None
        hidden_states_574 = torch.nn.functional.layer_norm(
            layer_output_37,
            (320,),
            l_self_modules_encoder_modules_layer_norm_modules_2_parameters_weight_,
            l_self_modules_encoder_modules_layer_norm_modules_2_parameters_bias_,
            1e-05,
        )
        layer_output_37 = (
            l_self_modules_encoder_modules_layer_norm_modules_2_parameters_weight_
        ) = l_self_modules_encoder_modules_layer_norm_modules_2_parameters_bias_ = None
        reshape_78 = hidden_states_574.reshape(1, 32, 32, -1)
        hidden_states_574 = None
        permute_116 = reshape_78.permute(0, 3, 1, 2)
        reshape_78 = None
        hidden_states_575 = permute_116.contiguous()
        permute_116 = None
        embeddings_9 = torch.conv2d(
            hidden_states_575,
            l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_weight_,
            l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_states_575 = l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_weight_ = l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_bias_ = (None)
        flatten_41 = embeddings_9.flatten(2)
        embeddings_9 = None
        embeddings_10 = flatten_41.transpose(1, 2)
        flatten_41 = None
        embeddings_11 = torch.nn.functional.layer_norm(
            embeddings_10,
            (512,),
            l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        embeddings_10 = l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_bias_ = (None)
        layer_norm_121 = torch.nn.functional.layer_norm(
            embeddings_11,
            (512,),
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_bias_ = (None)
        linear_228 = torch._C._nn.linear(
            layer_norm_121,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_190 = linear_228.view(1, -1, 8, 64)
        linear_228 = None
        query_layer_38 = view_190.transpose(1, 2)
        view_190 = None
        linear_229 = torch._C._nn.linear(
            layer_norm_121,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_191 = linear_229.view(1, -1, 8, 64)
        linear_229 = None
        key_layer_38 = view_191.transpose(1, 2)
        view_191 = None
        linear_230 = torch._C._nn.linear(
            layer_norm_121,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        layer_norm_121 = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_192 = linear_230.view(1, -1, 8, 64)
        linear_230 = None
        value_layer_38 = view_192.transpose(1, 2)
        view_192 = None
        transpose_235 = key_layer_38.transpose(-1, -2)
        key_layer_38 = None
        attention_scores_76 = torch.matmul(query_layer_38, transpose_235)
        query_layer_38 = transpose_235 = None
        attention_scores_77 = attention_scores_76 / 8.0
        attention_scores_76 = None
        attention_probs_76 = torch.nn.functional.softmax(attention_scores_77, dim=-1)
        attention_scores_77 = None
        attention_probs_77 = torch.nn.functional.dropout(
            attention_probs_76, 0.0, False, False
        )
        attention_probs_76 = None
        context_layer_114 = torch.matmul(attention_probs_77, value_layer_38)
        attention_probs_77 = value_layer_38 = None
        permute_117 = context_layer_114.permute(0, 2, 1, 3)
        context_layer_114 = None
        context_layer_115 = permute_117.contiguous()
        permute_117 = None
        context_layer_116 = context_layer_115.view((1, 256, 512))
        context_layer_115 = None
        hidden_states_576 = torch._C._nn.linear(
            context_layer_116,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_116 = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_577 = torch.nn.functional.dropout(
            hidden_states_576, 0.0, False, False
        )
        hidden_states_576 = None
        hidden_states_578 = hidden_states_577 + embeddings_11
        hidden_states_577 = embeddings_11 = None
        layer_norm_122 = torch.nn.functional.layer_norm(
            hidden_states_578,
            (512,),
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_579 = torch._C._nn.linear(
            layer_norm_122,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_122 = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_236 = hidden_states_579.transpose(1, 2)
        hidden_states_579 = None
        hidden_states_580 = transpose_236.view(1, 2048, 16, 16)
        transpose_236 = None
        hidden_states_581 = torch.conv2d(
            hidden_states_580,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        hidden_states_580 = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_42 = hidden_states_581.flatten(2)
        hidden_states_581 = None
        hidden_states_582 = flatten_42.transpose(1, 2)
        flatten_42 = None
        hidden_states_583 = torch._C._nn.gelu(hidden_states_582)
        hidden_states_582 = None
        hidden_states_584 = torch.nn.functional.dropout(
            hidden_states_583, 0.0, False, False
        )
        hidden_states_583 = None
        hidden_states_585 = torch._C._nn.linear(
            hidden_states_584,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_584 = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_586 = torch.nn.functional.dropout(
            hidden_states_585, 0.0, False, False
        )
        hidden_states_585 = None
        layer_output_38 = hidden_states_586 + hidden_states_578
        hidden_states_586 = hidden_states_578 = None
        layer_norm_123 = torch.nn.functional.layer_norm(
            layer_output_38,
            (512,),
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_bias_ = (None)
        linear_234 = torch._C._nn.linear(
            layer_norm_123,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_195 = linear_234.view(1, -1, 8, 64)
        linear_234 = None
        query_layer_39 = view_195.transpose(1, 2)
        view_195 = None
        linear_235 = torch._C._nn.linear(
            layer_norm_123,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_196 = linear_235.view(1, -1, 8, 64)
        linear_235 = None
        key_layer_39 = view_196.transpose(1, 2)
        view_196 = None
        linear_236 = torch._C._nn.linear(
            layer_norm_123,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        layer_norm_123 = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_197 = linear_236.view(1, -1, 8, 64)
        linear_236 = None
        value_layer_39 = view_197.transpose(1, 2)
        view_197 = None
        transpose_241 = key_layer_39.transpose(-1, -2)
        key_layer_39 = None
        attention_scores_78 = torch.matmul(query_layer_39, transpose_241)
        query_layer_39 = transpose_241 = None
        attention_scores_79 = attention_scores_78 / 8.0
        attention_scores_78 = None
        attention_probs_78 = torch.nn.functional.softmax(attention_scores_79, dim=-1)
        attention_scores_79 = None
        attention_probs_79 = torch.nn.functional.dropout(
            attention_probs_78, 0.0, False, False
        )
        attention_probs_78 = None
        context_layer_117 = torch.matmul(attention_probs_79, value_layer_39)
        attention_probs_79 = value_layer_39 = None
        permute_118 = context_layer_117.permute(0, 2, 1, 3)
        context_layer_117 = None
        context_layer_118 = permute_118.contiguous()
        permute_118 = None
        context_layer_119 = context_layer_118.view((1, 256, 512))
        context_layer_118 = None
        hidden_states_587 = torch._C._nn.linear(
            context_layer_119,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_119 = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_588 = torch.nn.functional.dropout(
            hidden_states_587, 0.0, False, False
        )
        hidden_states_587 = None
        hidden_states_589 = hidden_states_588 + layer_output_38
        hidden_states_588 = layer_output_38 = None
        layer_norm_124 = torch.nn.functional.layer_norm(
            hidden_states_589,
            (512,),
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_590 = torch._C._nn.linear(
            layer_norm_124,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_124 = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_242 = hidden_states_590.transpose(1, 2)
        hidden_states_590 = None
        hidden_states_591 = transpose_242.view(1, 2048, 16, 16)
        transpose_242 = None
        hidden_states_592 = torch.conv2d(
            hidden_states_591,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        hidden_states_591 = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_43 = hidden_states_592.flatten(2)
        hidden_states_592 = None
        hidden_states_593 = flatten_43.transpose(1, 2)
        flatten_43 = None
        hidden_states_594 = torch._C._nn.gelu(hidden_states_593)
        hidden_states_593 = None
        hidden_states_595 = torch.nn.functional.dropout(
            hidden_states_594, 0.0, False, False
        )
        hidden_states_594 = None
        hidden_states_596 = torch._C._nn.linear(
            hidden_states_595,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_595 = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_597 = torch.nn.functional.dropout(
            hidden_states_596, 0.0, False, False
        )
        hidden_states_596 = None
        layer_output_39 = hidden_states_597 + hidden_states_589
        hidden_states_597 = hidden_states_589 = None
        layer_norm_125 = torch.nn.functional.layer_norm(
            layer_output_39,
            (512,),
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_1_parameters_bias_ = (None)
        linear_240 = torch._C._nn.linear(
            layer_norm_125,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_200 = linear_240.view(1, -1, 8, 64)
        linear_240 = None
        query_layer_40 = view_200.transpose(1, 2)
        view_200 = None
        linear_241 = torch._C._nn.linear(
            layer_norm_125,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_201 = linear_241.view(1, -1, 8, 64)
        linear_241 = None
        key_layer_40 = view_201.transpose(1, 2)
        view_201 = None
        linear_242 = torch._C._nn.linear(
            layer_norm_125,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        layer_norm_125 = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_202 = linear_242.view(1, -1, 8, 64)
        linear_242 = None
        value_layer_40 = view_202.transpose(1, 2)
        view_202 = None
        transpose_247 = key_layer_40.transpose(-1, -2)
        key_layer_40 = None
        attention_scores_80 = torch.matmul(query_layer_40, transpose_247)
        query_layer_40 = transpose_247 = None
        attention_scores_81 = attention_scores_80 / 8.0
        attention_scores_80 = None
        attention_probs_80 = torch.nn.functional.softmax(attention_scores_81, dim=-1)
        attention_scores_81 = None
        attention_probs_81 = torch.nn.functional.dropout(
            attention_probs_80, 0.0, False, False
        )
        attention_probs_80 = None
        context_layer_120 = torch.matmul(attention_probs_81, value_layer_40)
        attention_probs_81 = value_layer_40 = None
        permute_119 = context_layer_120.permute(0, 2, 1, 3)
        context_layer_120 = None
        context_layer_121 = permute_119.contiguous()
        permute_119 = None
        context_layer_122 = context_layer_121.view((1, 256, 512))
        context_layer_121 = None
        hidden_states_598 = torch._C._nn.linear(
            context_layer_122,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_122 = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_599 = torch.nn.functional.dropout(
            hidden_states_598, 0.0, False, False
        )
        hidden_states_598 = None
        hidden_states_600 = hidden_states_599 + layer_output_39
        hidden_states_599 = layer_output_39 = None
        layer_norm_126 = torch.nn.functional.layer_norm(
            hidden_states_600,
            (512,),
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_601 = torch._C._nn.linear(
            layer_norm_126,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_126 = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_248 = hidden_states_601.transpose(1, 2)
        hidden_states_601 = None
        hidden_states_602 = transpose_248.view(1, 2048, 16, 16)
        transpose_248 = None
        hidden_states_603 = torch.conv2d(
            hidden_states_602,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        hidden_states_602 = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_44 = hidden_states_603.flatten(2)
        hidden_states_603 = None
        hidden_states_604 = flatten_44.transpose(1, 2)
        flatten_44 = None
        hidden_states_605 = torch._C._nn.gelu(hidden_states_604)
        hidden_states_604 = None
        hidden_states_606 = torch.nn.functional.dropout(
            hidden_states_605, 0.0, False, False
        )
        hidden_states_605 = None
        hidden_states_607 = torch._C._nn.linear(
            hidden_states_606,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_606 = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_608 = torch.nn.functional.dropout(
            hidden_states_607, 0.0, False, False
        )
        hidden_states_607 = None
        layer_output_40 = hidden_states_608 + hidden_states_600
        hidden_states_608 = hidden_states_600 = None
        hidden_states_609 = torch.nn.functional.layer_norm(
            layer_output_40,
            (512,),
            l_self_modules_encoder_modules_layer_norm_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layer_norm_modules_3_parameters_bias_,
            1e-05,
        )
        layer_output_40 = (
            l_self_modules_encoder_modules_layer_norm_modules_3_parameters_weight_
        ) = l_self_modules_encoder_modules_layer_norm_modules_3_parameters_bias_ = None
        reshape_79 = hidden_states_609.reshape(1, 16, 16, -1)
        hidden_states_609 = None
        permute_120 = reshape_79.permute(0, 3, 1, 2)
        reshape_79 = None
        hidden_states_610 = permute_120.contiguous()
        permute_120 = None
        return (hidden_states_610,)
