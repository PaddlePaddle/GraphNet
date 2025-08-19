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
        hidden_states_107 = torch.nn.functional.layer_norm(
            layer_output_6,
            (128,),
            l_self_modules_encoder_modules_layer_norm_modules_1_parameters_weight_,
            l_self_modules_encoder_modules_layer_norm_modules_1_parameters_bias_,
            1e-05,
        )
        layer_output_6 = (
            l_self_modules_encoder_modules_layer_norm_modules_1_parameters_weight_
        ) = l_self_modules_encoder_modules_layer_norm_modules_1_parameters_bias_ = None
        reshape_15 = hidden_states_107.reshape(1, 64, 64, -1)
        hidden_states_107 = None
        permute_22 = reshape_15.permute(0, 3, 1, 2)
        reshape_15 = None
        hidden_states_108 = permute_22.contiguous()
        permute_22 = None
        embeddings_6 = torch.conv2d(
            hidden_states_108,
            l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_weight_,
            l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_states_108 = l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_weight_ = l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_bias_ = (None)
        flatten_9 = embeddings_6.flatten(2)
        embeddings_6 = None
        embeddings_7 = flatten_9.transpose(1, 2)
        flatten_9 = None
        embeddings_8 = torch.nn.functional.layer_norm(
            embeddings_7,
            (320,),
            l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        embeddings_7 = l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_bias_ = (None)
        layer_norm_26 = torch.nn.functional.layer_norm(
            embeddings_8,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_bias_ = (None)
        linear_42 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_35 = linear_42.view(1, -1, 5, 64)
        linear_42 = None
        query_layer_7 = view_35.transpose(1, 2)
        view_35 = None
        permute_23 = layer_norm_26.permute(0, 2, 1)
        layer_norm_26 = None
        hidden_states_109 = permute_23.reshape(1, 320, 32, 32)
        permute_23 = None
        hidden_states_110 = torch.conv2d(
            hidden_states_109,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_109 = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_17 = hidden_states_110.reshape(1, 320, -1)
        hidden_states_110 = None
        hidden_states_111 = reshape_17.permute(0, 2, 1)
        reshape_17 = None
        hidden_states_112 = torch.nn.functional.layer_norm(
            hidden_states_111,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_111 = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_43 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_36 = linear_43.view(1, -1, 5, 64)
        linear_43 = None
        key_layer_7 = view_36.transpose(1, 2)
        view_36 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_112 = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_37 = linear_44.view(1, -1, 5, 64)
        linear_44 = None
        value_layer_7 = view_37.transpose(1, 2)
        view_37 = None
        transpose_48 = key_layer_7.transpose(-1, -2)
        key_layer_7 = None
        attention_scores_14 = torch.matmul(query_layer_7, transpose_48)
        query_layer_7 = transpose_48 = None
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
        permute_25 = context_layer_21.permute(0, 2, 1, 3)
        context_layer_21 = None
        context_layer_22 = permute_25.contiguous()
        permute_25 = None
        context_layer_23 = context_layer_22.view((1, 1024, 320))
        context_layer_22 = None
        hidden_states_113 = torch._C._nn.linear(
            context_layer_23,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_23 = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_114 = torch.nn.functional.dropout(
            hidden_states_113, 0.0, False, False
        )
        hidden_states_113 = None
        hidden_states_115 = hidden_states_114 + embeddings_8
        hidden_states_114 = embeddings_8 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            hidden_states_115,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_116 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_28 = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_49 = hidden_states_116.transpose(1, 2)
        hidden_states_116 = None
        hidden_states_117 = transpose_49.view(1, 1280, 32, 32)
        transpose_49 = None
        hidden_states_118 = torch.conv2d(
            hidden_states_117,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_117 = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_10 = hidden_states_118.flatten(2)
        hidden_states_118 = None
        hidden_states_119 = flatten_10.transpose(1, 2)
        flatten_10 = None
        hidden_states_120 = torch._C._nn.gelu(hidden_states_119)
        hidden_states_119 = None
        hidden_states_121 = torch.nn.functional.dropout(
            hidden_states_120, 0.0, False, False
        )
        hidden_states_120 = None
        hidden_states_122 = torch._C._nn.linear(
            hidden_states_121,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_121 = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_123 = torch.nn.functional.dropout(
            hidden_states_122, 0.0, False, False
        )
        hidden_states_122 = None
        layer_output_7 = hidden_states_123 + hidden_states_115
        hidden_states_123 = hidden_states_115 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            layer_output_7,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_bias_ = (None)
        linear_48 = torch._C._nn.linear(
            layer_norm_29,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_40 = linear_48.view(1, -1, 5, 64)
        linear_48 = None
        query_layer_8 = view_40.transpose(1, 2)
        view_40 = None
        permute_26 = layer_norm_29.permute(0, 2, 1)
        layer_norm_29 = None
        hidden_states_124 = permute_26.reshape(1, 320, 32, 32)
        permute_26 = None
        hidden_states_125 = torch.conv2d(
            hidden_states_124,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_124 = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_19 = hidden_states_125.reshape(1, 320, -1)
        hidden_states_125 = None
        hidden_states_126 = reshape_19.permute(0, 2, 1)
        reshape_19 = None
        hidden_states_127 = torch.nn.functional.layer_norm(
            hidden_states_126,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_126 = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_49 = torch._C._nn.linear(
            hidden_states_127,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_41 = linear_49.view(1, -1, 5, 64)
        linear_49 = None
        key_layer_8 = view_41.transpose(1, 2)
        view_41 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_127,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_127 = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_42 = linear_50.view(1, -1, 5, 64)
        linear_50 = None
        value_layer_8 = view_42.transpose(1, 2)
        view_42 = None
        transpose_54 = key_layer_8.transpose(-1, -2)
        key_layer_8 = None
        attention_scores_16 = torch.matmul(query_layer_8, transpose_54)
        query_layer_8 = transpose_54 = None
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
        permute_28 = context_layer_24.permute(0, 2, 1, 3)
        context_layer_24 = None
        context_layer_25 = permute_28.contiguous()
        permute_28 = None
        context_layer_26 = context_layer_25.view((1, 1024, 320))
        context_layer_25 = None
        hidden_states_128 = torch._C._nn.linear(
            context_layer_26,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_26 = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_129 = torch.nn.functional.dropout(
            hidden_states_128, 0.0, False, False
        )
        hidden_states_128 = None
        hidden_states_130 = hidden_states_129 + layer_output_7
        hidden_states_129 = layer_output_7 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            hidden_states_130,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_131 = torch._C._nn.linear(
            layer_norm_31,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_31 = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_55 = hidden_states_131.transpose(1, 2)
        hidden_states_131 = None
        hidden_states_132 = transpose_55.view(1, 1280, 32, 32)
        transpose_55 = None
        hidden_states_133 = torch.conv2d(
            hidden_states_132,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_132 = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_11 = hidden_states_133.flatten(2)
        hidden_states_133 = None
        hidden_states_134 = flatten_11.transpose(1, 2)
        flatten_11 = None
        hidden_states_135 = torch._C._nn.gelu(hidden_states_134)
        hidden_states_134 = None
        hidden_states_136 = torch.nn.functional.dropout(
            hidden_states_135, 0.0, False, False
        )
        hidden_states_135 = None
        hidden_states_137 = torch._C._nn.linear(
            hidden_states_136,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_136 = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_138 = torch.nn.functional.dropout(
            hidden_states_137, 0.0, False, False
        )
        hidden_states_137 = None
        layer_output_8 = hidden_states_138 + hidden_states_130
        hidden_states_138 = hidden_states_130 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            layer_output_8,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_1_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_45 = linear_54.view(1, -1, 5, 64)
        linear_54 = None
        query_layer_9 = view_45.transpose(1, 2)
        view_45 = None
        permute_29 = layer_norm_32.permute(0, 2, 1)
        layer_norm_32 = None
        hidden_states_139 = permute_29.reshape(1, 320, 32, 32)
        permute_29 = None
        hidden_states_140 = torch.conv2d(
            hidden_states_139,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_139 = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_21 = hidden_states_140.reshape(1, 320, -1)
        hidden_states_140 = None
        hidden_states_141 = reshape_21.permute(0, 2, 1)
        reshape_21 = None
        hidden_states_142 = torch.nn.functional.layer_norm(
            hidden_states_141,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_141 = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_55 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_46 = linear_55.view(1, -1, 5, 64)
        linear_55 = None
        key_layer_9 = view_46.transpose(1, 2)
        view_46 = None
        linear_56 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_142 = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_47 = linear_56.view(1, -1, 5, 64)
        linear_56 = None
        value_layer_9 = view_47.transpose(1, 2)
        view_47 = None
        transpose_60 = key_layer_9.transpose(-1, -2)
        key_layer_9 = None
        attention_scores_18 = torch.matmul(query_layer_9, transpose_60)
        query_layer_9 = transpose_60 = None
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
        permute_31 = context_layer_27.permute(0, 2, 1, 3)
        context_layer_27 = None
        context_layer_28 = permute_31.contiguous()
        permute_31 = None
        context_layer_29 = context_layer_28.view((1, 1024, 320))
        context_layer_28 = None
        hidden_states_143 = torch._C._nn.linear(
            context_layer_29,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_29 = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_144 = torch.nn.functional.dropout(
            hidden_states_143, 0.0, False, False
        )
        hidden_states_143 = None
        hidden_states_145 = hidden_states_144 + layer_output_8
        hidden_states_144 = layer_output_8 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            hidden_states_145,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_146 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_34 = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_61 = hidden_states_146.transpose(1, 2)
        hidden_states_146 = None
        hidden_states_147 = transpose_61.view(1, 1280, 32, 32)
        transpose_61 = None
        hidden_states_148 = torch.conv2d(
            hidden_states_147,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_147 = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_12 = hidden_states_148.flatten(2)
        hidden_states_148 = None
        hidden_states_149 = flatten_12.transpose(1, 2)
        flatten_12 = None
        hidden_states_150 = torch._C._nn.gelu(hidden_states_149)
        hidden_states_149 = None
        hidden_states_151 = torch.nn.functional.dropout(
            hidden_states_150, 0.0, False, False
        )
        hidden_states_150 = None
        hidden_states_152 = torch._C._nn.linear(
            hidden_states_151,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_151 = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_2_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_153 = torch.nn.functional.dropout(
            hidden_states_152, 0.0, False, False
        )
        hidden_states_152 = None
        layer_output_9 = hidden_states_153 + hidden_states_145
        hidden_states_153 = hidden_states_145 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            layer_output_9,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_1_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            layer_norm_35,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_50 = linear_60.view(1, -1, 5, 64)
        linear_60 = None
        query_layer_10 = view_50.transpose(1, 2)
        view_50 = None
        permute_32 = layer_norm_35.permute(0, 2, 1)
        layer_norm_35 = None
        hidden_states_154 = permute_32.reshape(1, 320, 32, 32)
        permute_32 = None
        hidden_states_155 = torch.conv2d(
            hidden_states_154,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_154 = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_23 = hidden_states_155.reshape(1, 320, -1)
        hidden_states_155 = None
        hidden_states_156 = reshape_23.permute(0, 2, 1)
        reshape_23 = None
        hidden_states_157 = torch.nn.functional.layer_norm(
            hidden_states_156,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_156 = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_61 = torch._C._nn.linear(
            hidden_states_157,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_51 = linear_61.view(1, -1, 5, 64)
        linear_61 = None
        key_layer_10 = view_51.transpose(1, 2)
        view_51 = None
        linear_62 = torch._C._nn.linear(
            hidden_states_157,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_157 = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_52 = linear_62.view(1, -1, 5, 64)
        linear_62 = None
        value_layer_10 = view_52.transpose(1, 2)
        view_52 = None
        transpose_66 = key_layer_10.transpose(-1, -2)
        key_layer_10 = None
        attention_scores_20 = torch.matmul(query_layer_10, transpose_66)
        query_layer_10 = transpose_66 = None
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
        permute_34 = context_layer_30.permute(0, 2, 1, 3)
        context_layer_30 = None
        context_layer_31 = permute_34.contiguous()
        permute_34 = None
        context_layer_32 = context_layer_31.view((1, 1024, 320))
        context_layer_31 = None
        hidden_states_158 = torch._C._nn.linear(
            context_layer_32,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_32 = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_159 = torch.nn.functional.dropout(
            hidden_states_158, 0.0, False, False
        )
        hidden_states_158 = None
        hidden_states_160 = hidden_states_159 + layer_output_9
        hidden_states_159 = layer_output_9 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            hidden_states_160,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_161 = torch._C._nn.linear(
            layer_norm_37,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_37 = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_67 = hidden_states_161.transpose(1, 2)
        hidden_states_161 = None
        hidden_states_162 = transpose_67.view(1, 1280, 32, 32)
        transpose_67 = None
        hidden_states_163 = torch.conv2d(
            hidden_states_162,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_162 = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_13 = hidden_states_163.flatten(2)
        hidden_states_163 = None
        hidden_states_164 = flatten_13.transpose(1, 2)
        flatten_13 = None
        hidden_states_165 = torch._C._nn.gelu(hidden_states_164)
        hidden_states_164 = None
        hidden_states_166 = torch.nn.functional.dropout(
            hidden_states_165, 0.0, False, False
        )
        hidden_states_165 = None
        hidden_states_167 = torch._C._nn.linear(
            hidden_states_166,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_166 = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_3_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_168 = torch.nn.functional.dropout(
            hidden_states_167, 0.0, False, False
        )
        hidden_states_167 = None
        layer_output_10 = hidden_states_168 + hidden_states_160
        hidden_states_168 = hidden_states_160 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            layer_output_10,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_1_parameters_bias_ = (None)
        linear_66 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_169 = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_25 = hidden_states_170.reshape(1, 320, -1)
        hidden_states_170 = None
        hidden_states_171 = reshape_25.permute(0, 2, 1)
        reshape_25 = None
        hidden_states_172 = torch.nn.functional.layer_norm(
            hidden_states_171,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_171 = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_67 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_56 = linear_67.view(1, -1, 5, 64)
        linear_67 = None
        key_layer_11 = view_56.transpose(1, 2)
        view_56 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_172 = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_35 = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_174 = torch.nn.functional.dropout(
            hidden_states_173, 0.0, False, False
        )
        hidden_states_173 = None
        hidden_states_175 = hidden_states_174 + layer_output_10
        hidden_states_174 = layer_output_10 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            hidden_states_175,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_176 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_40 = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_73 = hidden_states_176.transpose(1, 2)
        hidden_states_176 = None
        hidden_states_177 = transpose_73.view(1, 1280, 32, 32)
        transpose_73 = None
        hidden_states_178 = torch.conv2d(
            hidden_states_177,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_177 = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_181 = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_4_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_183 = torch.nn.functional.dropout(
            hidden_states_182, 0.0, False, False
        )
        hidden_states_182 = None
        layer_output_11 = hidden_states_183 + hidden_states_175
        hidden_states_183 = hidden_states_175 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            layer_output_11,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_1_parameters_bias_ = (None)
        linear_72 = torch._C._nn.linear(
            layer_norm_41,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_184 = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_27 = hidden_states_185.reshape(1, 320, -1)
        hidden_states_185 = None
        hidden_states_186 = reshape_27.permute(0, 2, 1)
        reshape_27 = None
        hidden_states_187 = torch.nn.functional.layer_norm(
            hidden_states_186,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_186 = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_73 = torch._C._nn.linear(
            hidden_states_187,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_61 = linear_73.view(1, -1, 5, 64)
        linear_73 = None
        key_layer_12 = view_61.transpose(1, 2)
        view_61 = None
        linear_74 = torch._C._nn.linear(
            hidden_states_187,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_187 = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_38 = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_189 = torch.nn.functional.dropout(
            hidden_states_188, 0.0, False, False
        )
        hidden_states_188 = None
        hidden_states_190 = hidden_states_189 + layer_output_11
        hidden_states_189 = layer_output_11 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            hidden_states_190,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_191 = torch._C._nn.linear(
            layer_norm_43,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_43 = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_79 = hidden_states_191.transpose(1, 2)
        hidden_states_191 = None
        hidden_states_192 = transpose_79.view(1, 1280, 32, 32)
        transpose_79 = None
        hidden_states_193 = torch.conv2d(
            hidden_states_192,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_192 = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_196 = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_5_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_198 = torch.nn.functional.dropout(
            hidden_states_197, 0.0, False, False
        )
        hidden_states_197 = None
        layer_output_12 = hidden_states_198 + hidden_states_190
        hidden_states_198 = hidden_states_190 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            layer_output_12,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_1_parameters_bias_ = (None)
        linear_78 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_199 = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_29 = hidden_states_200.reshape(1, 320, -1)
        hidden_states_200 = None
        hidden_states_201 = reshape_29.permute(0, 2, 1)
        reshape_29 = None
        hidden_states_202 = torch.nn.functional.layer_norm(
            hidden_states_201,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_201 = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_79 = torch._C._nn.linear(
            hidden_states_202,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_66 = linear_79.view(1, -1, 5, 64)
        linear_79 = None
        key_layer_13 = view_66.transpose(1, 2)
        view_66 = None
        linear_80 = torch._C._nn.linear(
            hidden_states_202,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_202 = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_41 = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_204 = torch.nn.functional.dropout(
            hidden_states_203, 0.0, False, False
        )
        hidden_states_203 = None
        hidden_states_205 = hidden_states_204 + layer_output_12
        hidden_states_204 = layer_output_12 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            hidden_states_205,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_206 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_46 = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_85 = hidden_states_206.transpose(1, 2)
        hidden_states_206 = None
        hidden_states_207 = transpose_85.view(1, 1280, 32, 32)
        transpose_85 = None
        hidden_states_208 = torch.conv2d(
            hidden_states_207,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_207 = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_211 = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_6_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_213 = torch.nn.functional.dropout(
            hidden_states_212, 0.0, False, False
        )
        hidden_states_212 = None
        layer_output_13 = hidden_states_213 + hidden_states_205
        hidden_states_213 = hidden_states_205 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            layer_output_13,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_1_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            layer_norm_47,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_214 = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_31 = hidden_states_215.reshape(1, 320, -1)
        hidden_states_215 = None
        hidden_states_216 = reshape_31.permute(0, 2, 1)
        reshape_31 = None
        hidden_states_217 = torch.nn.functional.layer_norm(
            hidden_states_216,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_216 = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_85 = torch._C._nn.linear(
            hidden_states_217,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_71 = linear_85.view(1, -1, 5, 64)
        linear_85 = None
        key_layer_14 = view_71.transpose(1, 2)
        view_71 = None
        linear_86 = torch._C._nn.linear(
            hidden_states_217,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_217 = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_44 = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_219 = torch.nn.functional.dropout(
            hidden_states_218, 0.0, False, False
        )
        hidden_states_218 = None
        hidden_states_220 = hidden_states_219 + layer_output_13
        hidden_states_219 = layer_output_13 = None
        layer_norm_49 = torch.nn.functional.layer_norm(
            hidden_states_220,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_221 = torch._C._nn.linear(
            layer_norm_49,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_49 = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_91 = hidden_states_221.transpose(1, 2)
        hidden_states_221 = None
        hidden_states_222 = transpose_91.view(1, 1280, 32, 32)
        transpose_91 = None
        hidden_states_223 = torch.conv2d(
            hidden_states_222,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_222 = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_226 = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_7_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_228 = torch.nn.functional.dropout(
            hidden_states_227, 0.0, False, False
        )
        hidden_states_227 = None
        layer_output_14 = hidden_states_228 + hidden_states_220
        hidden_states_228 = hidden_states_220 = None
        layer_norm_50 = torch.nn.functional.layer_norm(
            layer_output_14,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_1_parameters_bias_ = (None)
        linear_90 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_229 = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_33 = hidden_states_230.reshape(1, 320, -1)
        hidden_states_230 = None
        hidden_states_231 = reshape_33.permute(0, 2, 1)
        reshape_33 = None
        hidden_states_232 = torch.nn.functional.layer_norm(
            hidden_states_231,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_231 = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_91 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_76 = linear_91.view(1, -1, 5, 64)
        linear_91 = None
        key_layer_15 = view_76.transpose(1, 2)
        view_76 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_232 = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_47 = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_234 = torch.nn.functional.dropout(
            hidden_states_233, 0.0, False, False
        )
        hidden_states_233 = None
        hidden_states_235 = hidden_states_234 + layer_output_14
        hidden_states_234 = layer_output_14 = None
        layer_norm_52 = torch.nn.functional.layer_norm(
            hidden_states_235,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_236 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_52 = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_97 = hidden_states_236.transpose(1, 2)
        hidden_states_236 = None
        hidden_states_237 = transpose_97.view(1, 1280, 32, 32)
        transpose_97 = None
        hidden_states_238 = torch.conv2d(
            hidden_states_237,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_237 = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_241 = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_8_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_243 = torch.nn.functional.dropout(
            hidden_states_242, 0.0, False, False
        )
        hidden_states_242 = None
        layer_output_15 = hidden_states_243 + hidden_states_235
        hidden_states_243 = hidden_states_235 = None
        layer_norm_53 = torch.nn.functional.layer_norm(
            layer_output_15,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_1_parameters_bias_ = (None)
        linear_96 = torch._C._nn.linear(
            layer_norm_53,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_244 = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_35 = hidden_states_245.reshape(1, 320, -1)
        hidden_states_245 = None
        hidden_states_246 = reshape_35.permute(0, 2, 1)
        reshape_35 = None
        hidden_states_247 = torch.nn.functional.layer_norm(
            hidden_states_246,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_246 = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_97 = torch._C._nn.linear(
            hidden_states_247,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_81 = linear_97.view(1, -1, 5, 64)
        linear_97 = None
        key_layer_16 = view_81.transpose(1, 2)
        view_81 = None
        linear_98 = torch._C._nn.linear(
            hidden_states_247,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_247 = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_50 = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_249 = torch.nn.functional.dropout(
            hidden_states_248, 0.0, False, False
        )
        hidden_states_248 = None
        hidden_states_250 = hidden_states_249 + layer_output_15
        hidden_states_249 = layer_output_15 = None
        layer_norm_55 = torch.nn.functional.layer_norm(
            hidden_states_250,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_251 = torch._C._nn.linear(
            layer_norm_55,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_55 = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_103 = hidden_states_251.transpose(1, 2)
        hidden_states_251 = None
        hidden_states_252 = transpose_103.view(1, 1280, 32, 32)
        transpose_103 = None
        hidden_states_253 = torch.conv2d(
            hidden_states_252,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_252 = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_256 = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_9_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_258 = torch.nn.functional.dropout(
            hidden_states_257, 0.0, False, False
        )
        hidden_states_257 = None
        layer_output_16 = hidden_states_258 + hidden_states_250
        hidden_states_258 = hidden_states_250 = None
        layer_norm_56 = torch.nn.functional.layer_norm(
            layer_output_16,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_1_parameters_bias_ = (None)
        linear_102 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_259 = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_37 = hidden_states_260.reshape(1, 320, -1)
        hidden_states_260 = None
        hidden_states_261 = reshape_37.permute(0, 2, 1)
        reshape_37 = None
        hidden_states_262 = torch.nn.functional.layer_norm(
            hidden_states_261,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_261 = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_103 = torch._C._nn.linear(
            hidden_states_262,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_86 = linear_103.view(1, -1, 5, 64)
        linear_103 = None
        key_layer_17 = view_86.transpose(1, 2)
        view_86 = None
        linear_104 = torch._C._nn.linear(
            hidden_states_262,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_262 = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_53 = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_264 = torch.nn.functional.dropout(
            hidden_states_263, 0.0, False, False
        )
        hidden_states_263 = None
        hidden_states_265 = hidden_states_264 + layer_output_16
        hidden_states_264 = layer_output_16 = None
        layer_norm_58 = torch.nn.functional.layer_norm(
            hidden_states_265,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_266 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_58 = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_109 = hidden_states_266.transpose(1, 2)
        hidden_states_266 = None
        hidden_states_267 = transpose_109.view(1, 1280, 32, 32)
        transpose_109 = None
        hidden_states_268 = torch.conv2d(
            hidden_states_267,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_267 = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_271 = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_10_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_273 = torch.nn.functional.dropout(
            hidden_states_272, 0.0, False, False
        )
        hidden_states_272 = None
        layer_output_17 = hidden_states_273 + hidden_states_265
        hidden_states_273 = hidden_states_265 = None
        layer_norm_59 = torch.nn.functional.layer_norm(
            layer_output_17,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_1_parameters_bias_ = (None)
        linear_108 = torch._C._nn.linear(
            layer_norm_59,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_274 = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_39 = hidden_states_275.reshape(1, 320, -1)
        hidden_states_275 = None
        hidden_states_276 = reshape_39.permute(0, 2, 1)
        reshape_39 = None
        hidden_states_277 = torch.nn.functional.layer_norm(
            hidden_states_276,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_276 = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_109 = torch._C._nn.linear(
            hidden_states_277,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_91 = linear_109.view(1, -1, 5, 64)
        linear_109 = None
        key_layer_18 = view_91.transpose(1, 2)
        view_91 = None
        linear_110 = torch._C._nn.linear(
            hidden_states_277,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_277 = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_56 = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_279 = torch.nn.functional.dropout(
            hidden_states_278, 0.0, False, False
        )
        hidden_states_278 = None
        hidden_states_280 = hidden_states_279 + layer_output_17
        hidden_states_279 = layer_output_17 = None
        layer_norm_61 = torch.nn.functional.layer_norm(
            hidden_states_280,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_281 = torch._C._nn.linear(
            layer_norm_61,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_61 = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_115 = hidden_states_281.transpose(1, 2)
        hidden_states_281 = None
        hidden_states_282 = transpose_115.view(1, 1280, 32, 32)
        transpose_115 = None
        hidden_states_283 = torch.conv2d(
            hidden_states_282,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_282 = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_286 = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_11_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_288 = torch.nn.functional.dropout(
            hidden_states_287, 0.0, False, False
        )
        hidden_states_287 = None
        layer_output_18 = hidden_states_288 + hidden_states_280
        hidden_states_288 = hidden_states_280 = None
        layer_norm_62 = torch.nn.functional.layer_norm(
            layer_output_18,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_1_parameters_bias_ = (None)
        linear_114 = torch._C._nn.linear(
            layer_norm_62,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_289 = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_41 = hidden_states_290.reshape(1, 320, -1)
        hidden_states_290 = None
        hidden_states_291 = reshape_41.permute(0, 2, 1)
        reshape_41 = None
        hidden_states_292 = torch.nn.functional.layer_norm(
            hidden_states_291,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_291 = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_115 = torch._C._nn.linear(
            hidden_states_292,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_96 = linear_115.view(1, -1, 5, 64)
        linear_115 = None
        key_layer_19 = view_96.transpose(1, 2)
        view_96 = None
        linear_116 = torch._C._nn.linear(
            hidden_states_292,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_292 = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_59 = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_294 = torch.nn.functional.dropout(
            hidden_states_293, 0.0, False, False
        )
        hidden_states_293 = None
        hidden_states_295 = hidden_states_294 + layer_output_18
        hidden_states_294 = layer_output_18 = None
        layer_norm_64 = torch.nn.functional.layer_norm(
            hidden_states_295,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_296 = torch._C._nn.linear(
            layer_norm_64,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_64 = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_121 = hidden_states_296.transpose(1, 2)
        hidden_states_296 = None
        hidden_states_297 = transpose_121.view(1, 1280, 32, 32)
        transpose_121 = None
        hidden_states_298 = torch.conv2d(
            hidden_states_297,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_297 = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_301 = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_12_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_303 = torch.nn.functional.dropout(
            hidden_states_302, 0.0, False, False
        )
        hidden_states_302 = None
        layer_output_19 = hidden_states_303 + hidden_states_295
        hidden_states_303 = hidden_states_295 = None
        layer_norm_65 = torch.nn.functional.layer_norm(
            layer_output_19,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_1_parameters_bias_ = (None)
        linear_120 = torch._C._nn.linear(
            layer_norm_65,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_304 = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_43 = hidden_states_305.reshape(1, 320, -1)
        hidden_states_305 = None
        hidden_states_306 = reshape_43.permute(0, 2, 1)
        reshape_43 = None
        hidden_states_307 = torch.nn.functional.layer_norm(
            hidden_states_306,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_306 = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_121 = torch._C._nn.linear(
            hidden_states_307,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_101 = linear_121.view(1, -1, 5, 64)
        linear_121 = None
        key_layer_20 = view_101.transpose(1, 2)
        view_101 = None
        linear_122 = torch._C._nn.linear(
            hidden_states_307,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_307 = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_62 = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_309 = torch.nn.functional.dropout(
            hidden_states_308, 0.0, False, False
        )
        hidden_states_308 = None
        hidden_states_310 = hidden_states_309 + layer_output_19
        hidden_states_309 = layer_output_19 = None
        layer_norm_67 = torch.nn.functional.layer_norm(
            hidden_states_310,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_311 = torch._C._nn.linear(
            layer_norm_67,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_67 = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_127 = hidden_states_311.transpose(1, 2)
        hidden_states_311 = None
        hidden_states_312 = transpose_127.view(1, 1280, 32, 32)
        transpose_127 = None
        hidden_states_313 = torch.conv2d(
            hidden_states_312,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_312 = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_316 = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_13_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_318 = torch.nn.functional.dropout(
            hidden_states_317, 0.0, False, False
        )
        hidden_states_317 = None
        layer_output_20 = hidden_states_318 + hidden_states_310
        hidden_states_318 = hidden_states_310 = None
        layer_norm_68 = torch.nn.functional.layer_norm(
            layer_output_20,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_1_parameters_bias_ = (None)
        linear_126 = torch._C._nn.linear(
            layer_norm_68,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_319 = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_45 = hidden_states_320.reshape(1, 320, -1)
        hidden_states_320 = None
        hidden_states_321 = reshape_45.permute(0, 2, 1)
        reshape_45 = None
        hidden_states_322 = torch.nn.functional.layer_norm(
            hidden_states_321,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_321 = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_127 = torch._C._nn.linear(
            hidden_states_322,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_106 = linear_127.view(1, -1, 5, 64)
        linear_127 = None
        key_layer_21 = view_106.transpose(1, 2)
        view_106 = None
        linear_128 = torch._C._nn.linear(
            hidden_states_322,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_322 = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_65 = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_324 = torch.nn.functional.dropout(
            hidden_states_323, 0.0, False, False
        )
        hidden_states_323 = None
        hidden_states_325 = hidden_states_324 + layer_output_20
        hidden_states_324 = layer_output_20 = None
        layer_norm_70 = torch.nn.functional.layer_norm(
            hidden_states_325,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_326 = torch._C._nn.linear(
            layer_norm_70,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_70 = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_133 = hidden_states_326.transpose(1, 2)
        hidden_states_326 = None
        hidden_states_327 = transpose_133.view(1, 1280, 32, 32)
        transpose_133 = None
        hidden_states_328 = torch.conv2d(
            hidden_states_327,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_327 = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_331 = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_14_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_333 = torch.nn.functional.dropout(
            hidden_states_332, 0.0, False, False
        )
        hidden_states_332 = None
        layer_output_21 = hidden_states_333 + hidden_states_325
        hidden_states_333 = hidden_states_325 = None
        layer_norm_71 = torch.nn.functional.layer_norm(
            layer_output_21,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_1_parameters_bias_ = (None)
        linear_132 = torch._C._nn.linear(
            layer_norm_71,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_334 = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_47 = hidden_states_335.reshape(1, 320, -1)
        hidden_states_335 = None
        hidden_states_336 = reshape_47.permute(0, 2, 1)
        reshape_47 = None
        hidden_states_337 = torch.nn.functional.layer_norm(
            hidden_states_336,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_336 = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_133 = torch._C._nn.linear(
            hidden_states_337,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_111 = linear_133.view(1, -1, 5, 64)
        linear_133 = None
        key_layer_22 = view_111.transpose(1, 2)
        view_111 = None
        linear_134 = torch._C._nn.linear(
            hidden_states_337,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_337 = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_68 = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_339 = torch.nn.functional.dropout(
            hidden_states_338, 0.0, False, False
        )
        hidden_states_338 = None
        hidden_states_340 = hidden_states_339 + layer_output_21
        hidden_states_339 = layer_output_21 = None
        layer_norm_73 = torch.nn.functional.layer_norm(
            hidden_states_340,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_341 = torch._C._nn.linear(
            layer_norm_73,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_73 = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_139 = hidden_states_341.transpose(1, 2)
        hidden_states_341 = None
        hidden_states_342 = transpose_139.view(1, 1280, 32, 32)
        transpose_139 = None
        hidden_states_343 = torch.conv2d(
            hidden_states_342,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_342 = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_346 = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_15_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_348 = torch.nn.functional.dropout(
            hidden_states_347, 0.0, False, False
        )
        hidden_states_347 = None
        layer_output_22 = hidden_states_348 + hidden_states_340
        hidden_states_348 = hidden_states_340 = None
        layer_norm_74 = torch.nn.functional.layer_norm(
            layer_output_22,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_1_parameters_bias_ = (None)
        linear_138 = torch._C._nn.linear(
            layer_norm_74,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_349 = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_49 = hidden_states_350.reshape(1, 320, -1)
        hidden_states_350 = None
        hidden_states_351 = reshape_49.permute(0, 2, 1)
        reshape_49 = None
        hidden_states_352 = torch.nn.functional.layer_norm(
            hidden_states_351,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_351 = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_139 = torch._C._nn.linear(
            hidden_states_352,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_116 = linear_139.view(1, -1, 5, 64)
        linear_139 = None
        key_layer_23 = view_116.transpose(1, 2)
        view_116 = None
        linear_140 = torch._C._nn.linear(
            hidden_states_352,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_352 = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_71 = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_354 = torch.nn.functional.dropout(
            hidden_states_353, 0.0, False, False
        )
        hidden_states_353 = None
        hidden_states_355 = hidden_states_354 + layer_output_22
        hidden_states_354 = layer_output_22 = None
        layer_norm_76 = torch.nn.functional.layer_norm(
            hidden_states_355,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_356 = torch._C._nn.linear(
            layer_norm_76,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_76 = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_145 = hidden_states_356.transpose(1, 2)
        hidden_states_356 = None
        hidden_states_357 = transpose_145.view(1, 1280, 32, 32)
        transpose_145 = None
        hidden_states_358 = torch.conv2d(
            hidden_states_357,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_357 = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_361 = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_16_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_363 = torch.nn.functional.dropout(
            hidden_states_362, 0.0, False, False
        )
        hidden_states_362 = None
        layer_output_23 = hidden_states_363 + hidden_states_355
        hidden_states_363 = hidden_states_355 = None
        layer_norm_77 = torch.nn.functional.layer_norm(
            layer_output_23,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_1_parameters_bias_ = (None)
        linear_144 = torch._C._nn.linear(
            layer_norm_77,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_sr_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_sr_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_states_364 = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_sr_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_sr_parameters_bias_ = (None)
        reshape_51 = hidden_states_365.reshape(1, 320, -1)
        hidden_states_365 = None
        hidden_states_366 = reshape_51.permute(0, 2, 1)
        reshape_51 = None
        hidden_states_367 = torch.nn.functional.layer_norm(
            hidden_states_366,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_366 = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_layer_norm_parameters_bias_ = (None)
        linear_145 = torch._C._nn.linear(
            hidden_states_367,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_121 = linear_145.view(1, -1, 5, 64)
        linear_145 = None
        key_layer_24 = view_121.transpose(1, 2)
        view_121 = None
        linear_146 = torch._C._nn.linear(
            hidden_states_367,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        hidden_states_367 = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_74 = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_369 = torch.nn.functional.dropout(
            hidden_states_368, 0.0, False, False
        )
        hidden_states_368 = None
        hidden_states_370 = hidden_states_369 + layer_output_23
        hidden_states_369 = layer_output_23 = None
        layer_norm_79 = torch.nn.functional.layer_norm(
            hidden_states_370,
            (320,),
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_371 = torch._C._nn.linear(
            layer_norm_79,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_79 = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_151 = hidden_states_371.transpose(1, 2)
        hidden_states_371 = None
        hidden_states_372 = transpose_151.view(1, 1280, 32, 32)
        transpose_151 = None
        hidden_states_373 = torch.conv2d(
            hidden_states_372,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        hidden_states_372 = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
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
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_376 = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_2_modules_17_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_378 = torch.nn.functional.dropout(
            hidden_states_377, 0.0, False, False
        )
        hidden_states_377 = None
        layer_output_24 = hidden_states_378 + hidden_states_370
        hidden_states_378 = hidden_states_370 = None
        hidden_states_379 = torch.nn.functional.layer_norm(
            layer_output_24,
            (320,),
            l_self_modules_encoder_modules_layer_norm_modules_2_parameters_weight_,
            l_self_modules_encoder_modules_layer_norm_modules_2_parameters_bias_,
            1e-05,
        )
        layer_output_24 = (
            l_self_modules_encoder_modules_layer_norm_modules_2_parameters_weight_
        ) = l_self_modules_encoder_modules_layer_norm_modules_2_parameters_bias_ = None
        reshape_52 = hidden_states_379.reshape(1, 32, 32, -1)
        hidden_states_379 = None
        permute_77 = reshape_52.permute(0, 3, 1, 2)
        reshape_52 = None
        hidden_states_380 = permute_77.contiguous()
        permute_77 = None
        embeddings_9 = torch.conv2d(
            hidden_states_380,
            l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_weight_,
            l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_states_380 = l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_weight_ = l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_bias_ = (None)
        flatten_28 = embeddings_9.flatten(2)
        embeddings_9 = None
        embeddings_10 = flatten_28.transpose(1, 2)
        flatten_28 = None
        embeddings_11 = torch.nn.functional.layer_norm(
            embeddings_10,
            (512,),
            l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_bias_,
            1e-05,
        )
        embeddings_10 = l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_bias_ = (None)
        layer_norm_82 = torch.nn.functional.layer_norm(
            embeddings_11,
            (512,),
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_bias_ = (None)
        linear_150 = torch._C._nn.linear(
            layer_norm_82,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_125 = linear_150.view(1, -1, 8, 64)
        linear_150 = None
        query_layer_25 = view_125.transpose(1, 2)
        view_125 = None
        linear_151 = torch._C._nn.linear(
            layer_norm_82,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_126 = linear_151.view(1, -1, 8, 64)
        linear_151 = None
        key_layer_25 = view_126.transpose(1, 2)
        view_126 = None
        linear_152 = torch._C._nn.linear(
            layer_norm_82,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        layer_norm_82 = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_127 = linear_152.view(1, -1, 8, 64)
        linear_152 = None
        value_layer_25 = view_127.transpose(1, 2)
        view_127 = None
        transpose_157 = key_layer_25.transpose(-1, -2)
        key_layer_25 = None
        attention_scores_50 = torch.matmul(query_layer_25, transpose_157)
        query_layer_25 = transpose_157 = None
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
        permute_78 = context_layer_75.permute(0, 2, 1, 3)
        context_layer_75 = None
        context_layer_76 = permute_78.contiguous()
        permute_78 = None
        context_layer_77 = context_layer_76.view((1, 256, 512))
        context_layer_76 = None
        hidden_states_381 = torch._C._nn.linear(
            context_layer_77,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_77 = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_382 = torch.nn.functional.dropout(
            hidden_states_381, 0.0, False, False
        )
        hidden_states_381 = None
        hidden_states_383 = hidden_states_382 + embeddings_11
        hidden_states_382 = embeddings_11 = None
        layer_norm_83 = torch.nn.functional.layer_norm(
            hidden_states_383,
            (512,),
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_384 = torch._C._nn.linear(
            layer_norm_83,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_83 = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_158 = hidden_states_384.transpose(1, 2)
        hidden_states_384 = None
        hidden_states_385 = transpose_158.view(1, 2048, 16, 16)
        transpose_158 = None
        hidden_states_386 = torch.conv2d(
            hidden_states_385,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        hidden_states_385 = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_29 = hidden_states_386.flatten(2)
        hidden_states_386 = None
        hidden_states_387 = flatten_29.transpose(1, 2)
        flatten_29 = None
        hidden_states_388 = torch._C._nn.gelu(hidden_states_387)
        hidden_states_387 = None
        hidden_states_389 = torch.nn.functional.dropout(
            hidden_states_388, 0.0, False, False
        )
        hidden_states_388 = None
        hidden_states_390 = torch._C._nn.linear(
            hidden_states_389,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_389 = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_391 = torch.nn.functional.dropout(
            hidden_states_390, 0.0, False, False
        )
        hidden_states_390 = None
        layer_output_25 = hidden_states_391 + hidden_states_383
        hidden_states_391 = hidden_states_383 = None
        layer_norm_84 = torch.nn.functional.layer_norm(
            layer_output_25,
            (512,),
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_bias_ = (None)
        linear_156 = torch._C._nn.linear(
            layer_norm_84,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_130 = linear_156.view(1, -1, 8, 64)
        linear_156 = None
        query_layer_26 = view_130.transpose(1, 2)
        view_130 = None
        linear_157 = torch._C._nn.linear(
            layer_norm_84,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_131 = linear_157.view(1, -1, 8, 64)
        linear_157 = None
        key_layer_26 = view_131.transpose(1, 2)
        view_131 = None
        linear_158 = torch._C._nn.linear(
            layer_norm_84,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        layer_norm_84 = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_132 = linear_158.view(1, -1, 8, 64)
        linear_158 = None
        value_layer_26 = view_132.transpose(1, 2)
        view_132 = None
        transpose_163 = key_layer_26.transpose(-1, -2)
        key_layer_26 = None
        attention_scores_52 = torch.matmul(query_layer_26, transpose_163)
        query_layer_26 = transpose_163 = None
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
        permute_79 = context_layer_78.permute(0, 2, 1, 3)
        context_layer_78 = None
        context_layer_79 = permute_79.contiguous()
        permute_79 = None
        context_layer_80 = context_layer_79.view((1, 256, 512))
        context_layer_79 = None
        hidden_states_392 = torch._C._nn.linear(
            context_layer_80,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_80 = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_393 = torch.nn.functional.dropout(
            hidden_states_392, 0.0, False, False
        )
        hidden_states_392 = None
        hidden_states_394 = hidden_states_393 + layer_output_25
        hidden_states_393 = layer_output_25 = None
        layer_norm_85 = torch.nn.functional.layer_norm(
            hidden_states_394,
            (512,),
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_395 = torch._C._nn.linear(
            layer_norm_85,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_85 = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_164 = hidden_states_395.transpose(1, 2)
        hidden_states_395 = None
        hidden_states_396 = transpose_164.view(1, 2048, 16, 16)
        transpose_164 = None
        hidden_states_397 = torch.conv2d(
            hidden_states_396,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        hidden_states_396 = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_30 = hidden_states_397.flatten(2)
        hidden_states_397 = None
        hidden_states_398 = flatten_30.transpose(1, 2)
        flatten_30 = None
        hidden_states_399 = torch._C._nn.gelu(hidden_states_398)
        hidden_states_398 = None
        hidden_states_400 = torch.nn.functional.dropout(
            hidden_states_399, 0.0, False, False
        )
        hidden_states_399 = None
        hidden_states_401 = torch._C._nn.linear(
            hidden_states_400,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_400 = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_402 = torch.nn.functional.dropout(
            hidden_states_401, 0.0, False, False
        )
        hidden_states_401 = None
        layer_output_26 = hidden_states_402 + hidden_states_394
        hidden_states_402 = hidden_states_394 = None
        layer_norm_86 = torch.nn.functional.layer_norm(
            layer_output_26,
            (512,),
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_1_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_1_parameters_bias_ = (None)
        linear_162 = torch._C._nn.linear(
            layer_norm_86,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_query_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_query_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_query_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_query_parameters_bias_ = (None)
        view_135 = linear_162.view(1, -1, 8, 64)
        linear_162 = None
        query_layer_27 = view_135.transpose(1, 2)
        view_135 = None
        linear_163 = torch._C._nn.linear(
            layer_norm_86,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_key_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_key_parameters_bias_,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_key_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_key_parameters_bias_ = (None)
        view_136 = linear_163.view(1, -1, 8, 64)
        linear_163 = None
        key_layer_27 = view_136.transpose(1, 2)
        view_136 = None
        linear_164 = torch._C._nn.linear(
            layer_norm_86,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_value_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_value_parameters_bias_,
        )
        layer_norm_86 = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_value_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_self_modules_value_parameters_bias_ = (None)
        view_137 = linear_164.view(1, -1, 8, 64)
        linear_164 = None
        value_layer_27 = view_137.transpose(1, 2)
        view_137 = None
        transpose_169 = key_layer_27.transpose(-1, -2)
        key_layer_27 = None
        attention_scores_54 = torch.matmul(query_layer_27, transpose_169)
        query_layer_27 = transpose_169 = None
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
        permute_80 = context_layer_81.permute(0, 2, 1, 3)
        context_layer_81 = None
        context_layer_82 = permute_80.contiguous()
        permute_80 = None
        context_layer_83 = context_layer_82.view((1, 256, 512))
        context_layer_82 = None
        hidden_states_403 = torch._C._nn.linear(
            context_layer_83,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_,
        )
        context_layer_83 = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_ = (None)
        hidden_states_404 = torch.nn.functional.dropout(
            hidden_states_403, 0.0, False, False
        )
        hidden_states_403 = None
        hidden_states_405 = hidden_states_404 + layer_output_26
        hidden_states_404 = layer_output_26 = None
        layer_norm_87 = torch.nn.functional.layer_norm(
            hidden_states_405,
            (512,),
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_2_parameters_bias_,
            1e-05,
        )
        l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_layer_norm_2_parameters_bias_ = (None)
        hidden_states_406 = torch._C._nn.linear(
            layer_norm_87,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense1_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense1_parameters_bias_,
        )
        layer_norm_87 = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense1_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense1_parameters_bias_ = (None)
        transpose_170 = hidden_states_406.transpose(1, 2)
        hidden_states_406 = None
        hidden_states_407 = transpose_170.view(1, 2048, 16, 16)
        transpose_170 = None
        hidden_states_408 = torch.conv2d(
            hidden_states_407,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        hidden_states_407 = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_ = (None)
        flatten_31 = hidden_states_408.flatten(2)
        hidden_states_408 = None
        hidden_states_409 = flatten_31.transpose(1, 2)
        flatten_31 = None
        hidden_states_410 = torch._C._nn.gelu(hidden_states_409)
        hidden_states_409 = None
        hidden_states_411 = torch.nn.functional.dropout(
            hidden_states_410, 0.0, False, False
        )
        hidden_states_410 = None
        hidden_states_412 = torch._C._nn.linear(
            hidden_states_411,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense2_parameters_weight_,
            l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense2_parameters_bias_,
        )
        hidden_states_411 = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense2_parameters_weight_ = l_self_modules_encoder_modules_block_modules_3_modules_2_modules_mlp_modules_dense2_parameters_bias_ = (None)
        hidden_states_413 = torch.nn.functional.dropout(
            hidden_states_412, 0.0, False, False
        )
        hidden_states_412 = None
        layer_output_27 = hidden_states_413 + hidden_states_405
        hidden_states_413 = hidden_states_405 = None
        hidden_states_414 = torch.nn.functional.layer_norm(
            layer_output_27,
            (512,),
            l_self_modules_encoder_modules_layer_norm_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layer_norm_modules_3_parameters_bias_,
            1e-05,
        )
        layer_output_27 = (
            l_self_modules_encoder_modules_layer_norm_modules_3_parameters_weight_
        ) = l_self_modules_encoder_modules_layer_norm_modules_3_parameters_bias_ = None
        reshape_53 = hidden_states_414.reshape(1, 16, 16, -1)
        hidden_states_414 = None
        permute_81 = reshape_53.permute(0, 3, 1, 2)
        reshape_53 = None
        hidden_states_415 = permute_81.contiguous()
        permute_81 = None
        return (hidden_states_415,)
