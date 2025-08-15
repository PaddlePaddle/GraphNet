import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_cls_token1_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_cls_token2_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed3_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed3_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_cls_token3_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed4_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed4_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed4_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed4_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_cls_token4_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_norm12_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_norm12_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_norm13_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_norm13_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_norm14_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_norm14_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_norm22_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_norm22_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_norm23_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_norm23_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_norm24_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_norm24_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_norm12_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_norm12_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_norm13_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_norm13_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_norm14_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_norm14_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_norm22_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_norm22_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_norm23_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_norm23_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_norm24_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_norm24_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_norm12_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_norm12_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_norm13_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_norm13_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_norm14_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_norm14_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_norm22_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_norm22_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_norm23_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_norm23_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_norm24_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_norm24_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_norm12_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_norm12_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_norm13_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_norm13_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_norm14_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_norm14_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_norm22_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_norm22_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_norm23_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_norm23_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_norm24_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_norm24_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_norm12_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_norm12_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_norm13_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_norm13_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_norm14_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_norm14_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_norm22_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_norm22_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_norm23_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_norm23_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_norm24_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_norm24_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_norm12_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_norm12_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_norm13_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_norm13_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_norm14_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_norm14_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_norm22_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_norm22_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_norm23_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_norm23_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_norm24_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_norm24_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_aggregate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_aggregate_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_patch_embed1_modules_proj_parameters_weight_ = (
            L_self_modules_patch_embed1_modules_proj_parameters_weight_
        )
        l_self_modules_patch_embed1_modules_proj_parameters_bias_ = (
            L_self_modules_patch_embed1_modules_proj_parameters_bias_
        )
        l_self_modules_patch_embed1_modules_norm_parameters_weight_ = (
            L_self_modules_patch_embed1_modules_norm_parameters_weight_
        )
        l_self_modules_patch_embed1_modules_norm_parameters_bias_ = (
            L_self_modules_patch_embed1_modules_norm_parameters_bias_
        )
        l_self_parameters_cls_token1_ = L_self_parameters_cls_token1_
        l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks1_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks1_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks1_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks1_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_
        l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_
        l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_
        l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_
        l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_
        l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_
        l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks1_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks1_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks1_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks1_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks1_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks1_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks1_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks1_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks1_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks1_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks1_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks1_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_patch_embed2_modules_proj_parameters_weight_ = (
            L_self_modules_patch_embed2_modules_proj_parameters_weight_
        )
        l_self_modules_patch_embed2_modules_proj_parameters_bias_ = (
            L_self_modules_patch_embed2_modules_proj_parameters_bias_
        )
        l_self_modules_patch_embed2_modules_norm_parameters_weight_ = (
            L_self_modules_patch_embed2_modules_norm_parameters_weight_
        )
        l_self_modules_patch_embed2_modules_norm_parameters_bias_ = (
            L_self_modules_patch_embed2_modules_norm_parameters_bias_
        )
        l_self_parameters_cls_token2_ = L_self_parameters_cls_token2_
        l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_
        l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_
        l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_
        l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_
        l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_
        l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_
        l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_patch_embed3_modules_proj_parameters_weight_ = (
            L_self_modules_patch_embed3_modules_proj_parameters_weight_
        )
        l_self_modules_patch_embed3_modules_proj_parameters_bias_ = (
            L_self_modules_patch_embed3_modules_proj_parameters_bias_
        )
        l_self_modules_patch_embed3_modules_norm_parameters_weight_ = (
            L_self_modules_patch_embed3_modules_norm_parameters_weight_
        )
        l_self_modules_patch_embed3_modules_norm_parameters_bias_ = (
            L_self_modules_patch_embed3_modules_norm_parameters_bias_
        )
        l_self_parameters_cls_token3_ = L_self_parameters_cls_token3_
        l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_
        l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_
        l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_
        l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_
        l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_
        l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_
        l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_patch_embed4_modules_proj_parameters_weight_ = (
            L_self_modules_patch_embed4_modules_proj_parameters_weight_
        )
        l_self_modules_patch_embed4_modules_proj_parameters_bias_ = (
            L_self_modules_patch_embed4_modules_proj_parameters_bias_
        )
        l_self_modules_patch_embed4_modules_norm_parameters_weight_ = (
            L_self_modules_patch_embed4_modules_norm_parameters_weight_
        )
        l_self_modules_patch_embed4_modules_norm_parameters_bias_ = (
            L_self_modules_patch_embed4_modules_norm_parameters_bias_
        )
        l_self_parameters_cls_token4_ = L_self_parameters_cls_token4_
        l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_
        l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_
        l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_
        l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_
        l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_
        l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_
        l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_parallel_blocks_modules_0_modules_norm12_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_0_modules_norm12_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm12_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_0_modules_norm12_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm13_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_0_modules_norm13_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm13_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_0_modules_norm13_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm14_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_0_modules_norm14_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm14_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_0_modules_norm14_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_0_modules_norm22_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_0_modules_norm22_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm22_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_0_modules_norm22_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm23_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_0_modules_norm23_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm23_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_0_modules_norm23_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm24_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_0_modules_norm24_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm24_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_0_modules_norm24_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_weight_ = L_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_weight_
        l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_bias_ = L_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_bias_
        l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_weight_ = L_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_weight_
        l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_bias_ = L_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_bias_
        l_self_modules_parallel_blocks_modules_1_modules_norm12_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_1_modules_norm12_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm12_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_1_modules_norm12_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm13_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_1_modules_norm13_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm13_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_1_modules_norm13_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm14_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_1_modules_norm14_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm14_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_1_modules_norm14_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_1_modules_norm22_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_1_modules_norm22_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm22_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_1_modules_norm22_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm23_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_1_modules_norm23_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm23_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_1_modules_norm23_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm24_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_1_modules_norm24_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm24_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_1_modules_norm24_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_weight_ = L_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_weight_
        l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_bias_ = L_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_bias_
        l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_weight_ = L_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_weight_
        l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_bias_ = L_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_bias_
        l_self_modules_parallel_blocks_modules_2_modules_norm12_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_2_modules_norm12_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm12_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_2_modules_norm12_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm13_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_2_modules_norm13_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm13_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_2_modules_norm13_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm14_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_2_modules_norm14_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm14_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_2_modules_norm14_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_2_modules_norm22_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_2_modules_norm22_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm22_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_2_modules_norm22_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm23_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_2_modules_norm23_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm23_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_2_modules_norm23_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm24_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_2_modules_norm24_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm24_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_2_modules_norm24_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_weight_ = L_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_weight_
        l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_bias_ = L_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_bias_
        l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_weight_ = L_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_weight_
        l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_bias_ = L_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_bias_
        l_self_modules_parallel_blocks_modules_3_modules_norm12_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_3_modules_norm12_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm12_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_3_modules_norm12_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm13_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_3_modules_norm13_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm13_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_3_modules_norm13_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm14_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_3_modules_norm14_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm14_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_3_modules_norm14_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_3_modules_norm22_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_3_modules_norm22_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm22_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_3_modules_norm22_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm23_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_3_modules_norm23_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm23_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_3_modules_norm23_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm24_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_3_modules_norm24_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm24_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_3_modules_norm24_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_weight_ = L_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_weight_
        l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_bias_ = L_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_bias_
        l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_weight_ = L_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_weight_
        l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_bias_ = L_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_bias_
        l_self_modules_parallel_blocks_modules_4_modules_norm12_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_4_modules_norm12_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm12_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_4_modules_norm12_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm13_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_4_modules_norm13_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm13_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_4_modules_norm13_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm14_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_4_modules_norm14_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm14_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_4_modules_norm14_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_4_modules_norm22_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_4_modules_norm22_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm22_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_4_modules_norm22_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm23_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_4_modules_norm23_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm23_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_4_modules_norm23_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm24_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_4_modules_norm24_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm24_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_4_modules_norm24_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_weight_ = L_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_weight_
        l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_bias_ = L_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_bias_
        l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_weight_ = L_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_weight_
        l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_bias_ = L_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_bias_
        l_self_modules_parallel_blocks_modules_5_modules_norm12_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_5_modules_norm12_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm12_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_5_modules_norm12_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm13_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_5_modules_norm13_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm13_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_5_modules_norm13_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm14_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_5_modules_norm14_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm14_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_5_modules_norm14_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_qkv_parameters_weight_ = L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_qkv_parameters_weight_
        l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_qkv_parameters_bias_ = L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_qkv_parameters_bias_
        l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_proj_parameters_weight_ = L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_proj_parameters_weight_
        l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_proj_parameters_bias_ = L_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_proj_parameters_bias_
        l_self_modules_parallel_blocks_modules_5_modules_norm22_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_5_modules_norm22_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm22_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_5_modules_norm22_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm23_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_5_modules_norm23_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm23_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_5_modules_norm23_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm24_parameters_weight_ = (
            L_self_modules_parallel_blocks_modules_5_modules_norm24_parameters_weight_
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm24_parameters_bias_ = (
            L_self_modules_parallel_blocks_modules_5_modules_norm24_parameters_bias_
        )
        l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_weight_ = L_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_weight_
        l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_bias_ = L_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_bias_
        l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_weight_ = L_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_weight_
        l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_bias_ = L_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_bias_
        l_self_modules_norm2_parameters_weight_ = (
            L_self_modules_norm2_parameters_weight_
        )
        l_self_modules_norm2_parameters_bias_ = L_self_modules_norm2_parameters_bias_
        l_self_modules_norm3_parameters_weight_ = (
            L_self_modules_norm3_parameters_weight_
        )
        l_self_modules_norm3_parameters_bias_ = L_self_modules_norm3_parameters_bias_
        l_self_modules_norm4_parameters_weight_ = (
            L_self_modules_norm4_parameters_weight_
        )
        l_self_modules_norm4_parameters_bias_ = L_self_modules_norm4_parameters_bias_
        l_self_modules_aggregate_parameters_weight_ = (
            L_self_modules_aggregate_parameters_weight_
        )
        l_self_modules_aggregate_parameters_bias_ = (
            L_self_modules_aggregate_parameters_bias_
        )
        l_self_modules_head_parameters_weight_ = L_self_modules_head_parameters_weight_
        l_self_modules_head_parameters_bias_ = L_self_modules_head_parameters_bias_
        x = torch.conv2d(
            l_x_,
            l_self_modules_patch_embed1_modules_proj_parameters_weight_,
            l_self_modules_patch_embed1_modules_proj_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_patch_embed1_modules_proj_parameters_weight_
        ) = l_self_modules_patch_embed1_modules_proj_parameters_bias_ = None
        flatten = x.flatten(2)
        x = None
        x_1 = flatten.transpose(1, 2)
        flatten = None
        x_2 = torch.nn.functional.layer_norm(
            x_1,
            (152,),
            l_self_modules_patch_embed1_modules_norm_parameters_weight_,
            l_self_modules_patch_embed1_modules_norm_parameters_bias_,
            1e-05,
        )
        x_1 = (
            l_self_modules_patch_embed1_modules_norm_parameters_weight_
        ) = l_self_modules_patch_embed1_modules_norm_parameters_bias_ = None
        cls_tokens = l_self_parameters_cls_token1_.expand(1, -1, -1)
        l_self_parameters_cls_token1_ = None
        x_3 = torch.cat((cls_tokens, x_2), dim=1)
        cls_tokens = x_2 = None
        cls_token = x_3[(slice(None, None, None), slice(None, 1, None))]
        img_tokens = x_3[(slice(None, None, None), slice(1, None, None))]
        x_3 = None
        transpose_1 = img_tokens.transpose(1, 2)
        img_tokens = None
        feat = transpose_1.view(1, 152, 56, 56)
        transpose_1 = None
        conv2d_1 = torch.conv2d(
            feat,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_4 = conv2d_1 + feat
        conv2d_1 = feat = None
        flatten_1 = x_4.flatten(2)
        x_4 = None
        x_5 = flatten_1.transpose(1, 2)
        flatten_1 = None
        x_6 = torch.cat((cls_token, x_5), dim=1)
        cls_token = x_5 = None
        x_7 = torch.nn.functional.layer_norm(
            x_6,
            (152,),
            l_self_modules_serial_blocks1_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks1_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks1_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear = torch._C._nn.linear(
            x_7,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_7 = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape = linear.reshape(1, 3137, 3, 8, 19)
        linear = None
        qkv = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        unbind = qkv.unbind(0)
        qkv = None
        q = unbind[0]
        k = unbind[1]
        v = unbind[2]
        unbind = None
        k_softmax = k.softmax(dim=2)
        k = None
        transpose_3 = k_softmax.transpose(-1, -2)
        k_softmax = None
        factor_att = transpose_3 @ v
        transpose_3 = None
        factor_att_1 = q @ factor_att
        factor_att = None
        q_img = q[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q = None
        v_img = v[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v = None
        transpose_4 = v_img.transpose(-1, -2)
        v_img = None
        v_img_1 = transpose_4.reshape(1, 152, 56, 56)
        transpose_4 = None
        split = torch.functional.split(v_img_1, [38, 57, 57], dim=1)
        v_img_1 = None
        getitem_7 = split[0]
        getitem_8 = split[1]
        getitem_9 = split[2]
        split = None
        conv2d_2 = torch.conv2d(
            getitem_7,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_7 = None
        conv2d_3 = torch.conv2d(
            getitem_8,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_8 = None
        conv2d_4 = torch.conv2d(
            getitem_9,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_9 = None
        conv_v_img = torch.cat([conv2d_2, conv2d_3, conv2d_4], dim=1)
        conv2d_2 = conv2d_3 = conv2d_4 = None
        reshape_2 = conv_v_img.reshape(1, 8, 19, 3136)
        conv_v_img = None
        conv_v_img_1 = reshape_2.transpose(-1, -2)
        reshape_2 = None
        EV_hat = q_img * conv_v_img_1
        q_img = conv_v_img_1 = None
        EV_hat_1 = torch._C._nn.pad(EV_hat, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat = None
        mul_1 = 0.22941573387056177 * factor_att_1
        factor_att_1 = None
        x_8 = mul_1 + EV_hat_1
        mul_1 = EV_hat_1 = None
        transpose_6 = x_8.transpose(1, 2)
        x_8 = None
        x_9 = transpose_6.reshape(1, 3137, 152)
        transpose_6 = None
        x_10 = torch._C._nn.linear(
            x_9,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_9 = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_11 = torch.nn.functional.dropout(x_10, 0.0, False, False)
        x_10 = None
        x_12 = x_6 + x_11
        x_6 = x_11 = None
        x_13 = torch.nn.functional.layer_norm(
            x_12,
            (152,),
            l_self_modules_serial_blocks1_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks1_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks1_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_14 = torch._C._nn.linear(
            x_13,
            l_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_13 = l_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_15 = torch._C._nn.gelu(x_14, approximate="none")
        x_14 = None
        x_16 = torch.nn.functional.dropout(x_15, 0.0, False, False)
        x_15 = None
        x_17 = torch._C._nn.linear(
            x_16,
            l_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_16 = l_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_18 = torch.nn.functional.dropout(x_17, 0.0, False, False)
        x_17 = None
        x_19 = x_12 + x_18
        x_12 = x_18 = None
        cls_token_1 = x_19[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_1 = x_19[(slice(None, None, None), slice(1, None, None))]
        x_19 = None
        transpose_7 = img_tokens_1.transpose(1, 2)
        img_tokens_1 = None
        feat_1 = transpose_7.view(1, 152, 56, 56)
        transpose_7 = None
        conv2d_5 = torch.conv2d(
            feat_1,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_ = (None)
        x_20 = conv2d_5 + feat_1
        conv2d_5 = feat_1 = None
        flatten_2 = x_20.flatten(2)
        x_20 = None
        x_21 = flatten_2.transpose(1, 2)
        flatten_2 = None
        x_22 = torch.cat((cls_token_1, x_21), dim=1)
        cls_token_1 = x_21 = None
        x_23 = torch.nn.functional.layer_norm(
            x_22,
            (152,),
            l_self_modules_serial_blocks1_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks1_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks1_modules_1_modules_norm1_parameters_bias_
        ) = None
        linear_4 = torch._C._nn.linear(
            x_23,
            l_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_23 = l_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_4 = linear_4.reshape(1, 3137, 3, 8, 19)
        linear_4 = None
        qkv_1 = reshape_4.permute(2, 0, 3, 1, 4)
        reshape_4 = None
        unbind_1 = qkv_1.unbind(0)
        qkv_1 = None
        q_1 = unbind_1[0]
        k_1 = unbind_1[1]
        v_1 = unbind_1[2]
        unbind_1 = None
        k_softmax_1 = k_1.softmax(dim=2)
        k_1 = None
        transpose_9 = k_softmax_1.transpose(-1, -2)
        k_softmax_1 = None
        factor_att_2 = transpose_9 @ v_1
        transpose_9 = None
        factor_att_3 = q_1 @ factor_att_2
        factor_att_2 = None
        q_img_1 = q_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_1 = None
        v_img_2 = v_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_1 = None
        transpose_10 = v_img_2.transpose(-1, -2)
        v_img_2 = None
        v_img_3 = transpose_10.reshape(1, 152, 56, 56)
        transpose_10 = None
        split_1 = torch.functional.split(v_img_3, [38, 57, 57], dim=1)
        v_img_3 = None
        getitem_17 = split_1[0]
        getitem_18 = split_1[1]
        getitem_19 = split_1[2]
        split_1 = None
        conv2d_6 = torch.conv2d(
            getitem_17,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_17 = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_7 = torch.conv2d(
            getitem_18,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_18 = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_8 = torch.conv2d(
            getitem_19,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_19 = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
        conv_v_img_2 = torch.cat([conv2d_6, conv2d_7, conv2d_8], dim=1)
        conv2d_6 = conv2d_7 = conv2d_8 = None
        reshape_6 = conv_v_img_2.reshape(1, 8, 19, 3136)
        conv_v_img_2 = None
        conv_v_img_3 = reshape_6.transpose(-1, -2)
        reshape_6 = None
        EV_hat_2 = q_img_1 * conv_v_img_3
        q_img_1 = conv_v_img_3 = None
        EV_hat_3 = torch._C._nn.pad(EV_hat_2, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_2 = None
        mul_3 = 0.22941573387056177 * factor_att_3
        factor_att_3 = None
        x_24 = mul_3 + EV_hat_3
        mul_3 = EV_hat_3 = None
        transpose_12 = x_24.transpose(1, 2)
        x_24 = None
        x_25 = transpose_12.reshape(1, 3137, 152)
        transpose_12 = None
        x_26 = torch._C._nn.linear(
            x_25,
            l_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_25 = l_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks1_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_27 = torch.nn.functional.dropout(x_26, 0.0, False, False)
        x_26 = None
        x_28 = x_22 + x_27
        x_22 = x_27 = None
        x_29 = torch.nn.functional.layer_norm(
            x_28,
            (152,),
            l_self_modules_serial_blocks1_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks1_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks1_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_30 = torch._C._nn.linear(
            x_29,
            l_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_29 = l_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_31 = torch._C._nn.gelu(x_30, approximate="none")
        x_30 = None
        x_32 = torch.nn.functional.dropout(x_31, 0.0, False, False)
        x_31 = None
        x_33 = torch._C._nn.linear(
            x_32,
            l_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_32 = l_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_34 = torch.nn.functional.dropout(x_33, 0.0, False, False)
        x_33 = None
        x_35 = x_28 + x_34
        x_28 = x_34 = None
        getitem_20 = x_35[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        x_35 = None
        reshape_8 = getitem_20.reshape(1, 56, 56, -1)
        getitem_20 = None
        permute_2 = reshape_8.permute(0, 3, 1, 2)
        reshape_8 = None
        x1_nocls = permute_2.contiguous()
        permute_2 = None
        x_36 = torch.conv2d(
            x1_nocls,
            l_self_modules_patch_embed2_modules_proj_parameters_weight_,
            l_self_modules_patch_embed2_modules_proj_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x1_nocls = (
            l_self_modules_patch_embed2_modules_proj_parameters_weight_
        ) = l_self_modules_patch_embed2_modules_proj_parameters_bias_ = None
        flatten_3 = x_36.flatten(2)
        x_36 = None
        x_37 = flatten_3.transpose(1, 2)
        flatten_3 = None
        x_38 = torch.nn.functional.layer_norm(
            x_37,
            (152,),
            l_self_modules_patch_embed2_modules_norm_parameters_weight_,
            l_self_modules_patch_embed2_modules_norm_parameters_bias_,
            1e-05,
        )
        x_37 = (
            l_self_modules_patch_embed2_modules_norm_parameters_weight_
        ) = l_self_modules_patch_embed2_modules_norm_parameters_bias_ = None
        cls_tokens_1 = l_self_parameters_cls_token2_.expand(1, -1, -1)
        l_self_parameters_cls_token2_ = None
        x_39 = torch.cat((cls_tokens_1, x_38), dim=1)
        cls_tokens_1 = x_38 = None
        cls_token_2 = x_39[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_2 = x_39[(slice(None, None, None), slice(1, None, None))]
        x_39 = None
        transpose_14 = img_tokens_2.transpose(1, 2)
        img_tokens_2 = None
        feat_2 = transpose_14.view(1, 152, 28, 28)
        transpose_14 = None
        conv2d_10 = torch.conv2d(
            feat_2,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_40 = conv2d_10 + feat_2
        conv2d_10 = feat_2 = None
        flatten_4 = x_40.flatten(2)
        x_40 = None
        x_41 = flatten_4.transpose(1, 2)
        flatten_4 = None
        x_42 = torch.cat((cls_token_2, x_41), dim=1)
        cls_token_2 = x_41 = None
        x_43 = torch.nn.functional.layer_norm(
            x_42,
            (152,),
            l_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear_8 = torch._C._nn.linear(
            x_43,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_43 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_9 = linear_8.reshape(1, 785, 3, 8, 19)
        linear_8 = None
        qkv_2 = reshape_9.permute(2, 0, 3, 1, 4)
        reshape_9 = None
        unbind_2 = qkv_2.unbind(0)
        qkv_2 = None
        q_2 = unbind_2[0]
        k_2 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        k_softmax_2 = k_2.softmax(dim=2)
        k_2 = None
        transpose_16 = k_softmax_2.transpose(-1, -2)
        k_softmax_2 = None
        factor_att_4 = transpose_16 @ v_2
        transpose_16 = None
        factor_att_5 = q_2 @ factor_att_4
        factor_att_4 = None
        q_img_2 = q_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_2 = None
        v_img_4 = v_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_2 = None
        transpose_17 = v_img_4.transpose(-1, -2)
        v_img_4 = None
        v_img_5 = transpose_17.reshape(1, 152, 28, 28)
        transpose_17 = None
        split_2 = torch.functional.split(v_img_5, [38, 57, 57], dim=1)
        v_img_5 = None
        getitem_28 = split_2[0]
        getitem_29 = split_2[1]
        getitem_30 = split_2[2]
        split_2 = None
        conv2d_11 = torch.conv2d(
            getitem_28,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_28 = None
        conv2d_12 = torch.conv2d(
            getitem_29,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_29 = None
        conv2d_13 = torch.conv2d(
            getitem_30,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_30 = None
        conv_v_img_4 = torch.cat([conv2d_11, conv2d_12, conv2d_13], dim=1)
        conv2d_11 = conv2d_12 = conv2d_13 = None
        reshape_11 = conv_v_img_4.reshape(1, 8, 19, 784)
        conv_v_img_4 = None
        conv_v_img_5 = reshape_11.transpose(-1, -2)
        reshape_11 = None
        EV_hat_4 = q_img_2 * conv_v_img_5
        q_img_2 = conv_v_img_5 = None
        EV_hat_5 = torch._C._nn.pad(EV_hat_4, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_4 = None
        mul_5 = 0.22941573387056177 * factor_att_5
        factor_att_5 = None
        x_44 = mul_5 + EV_hat_5
        mul_5 = EV_hat_5 = None
        transpose_19 = x_44.transpose(1, 2)
        x_44 = None
        x_45 = transpose_19.reshape(1, 785, 152)
        transpose_19 = None
        x_46 = torch._C._nn.linear(
            x_45,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_45 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_47 = torch.nn.functional.dropout(x_46, 0.0, False, False)
        x_46 = None
        x_48 = x_42 + x_47
        x_42 = x_47 = None
        x_49 = torch.nn.functional.layer_norm(
            x_48,
            (152,),
            l_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_50 = torch._C._nn.linear(
            x_49,
            l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_49 = l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_51 = torch._C._nn.gelu(x_50, approximate="none")
        x_50 = None
        x_52 = torch.nn.functional.dropout(x_51, 0.0, False, False)
        x_51 = None
        x_53 = torch._C._nn.linear(
            x_52,
            l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_52 = l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_54 = torch.nn.functional.dropout(x_53, 0.0, False, False)
        x_53 = None
        x_55 = x_48 + x_54
        x_48 = x_54 = None
        cls_token_3 = x_55[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_3 = x_55[(slice(None, None, None), slice(1, None, None))]
        x_55 = None
        transpose_20 = img_tokens_3.transpose(1, 2)
        img_tokens_3 = None
        feat_3 = transpose_20.view(1, 152, 28, 28)
        transpose_20 = None
        conv2d_14 = torch.conv2d(
            feat_3,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_56 = conv2d_14 + feat_3
        conv2d_14 = feat_3 = None
        flatten_5 = x_56.flatten(2)
        x_56 = None
        x_57 = flatten_5.transpose(1, 2)
        flatten_5 = None
        x_58 = torch.cat((cls_token_3, x_57), dim=1)
        cls_token_3 = x_57 = None
        x_59 = torch.nn.functional.layer_norm(
            x_58,
            (152,),
            l_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_bias_
        ) = None
        linear_12 = torch._C._nn.linear(
            x_59,
            l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_59 = l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_13 = linear_12.reshape(1, 785, 3, 8, 19)
        linear_12 = None
        qkv_3 = reshape_13.permute(2, 0, 3, 1, 4)
        reshape_13 = None
        unbind_3 = qkv_3.unbind(0)
        qkv_3 = None
        q_3 = unbind_3[0]
        k_3 = unbind_3[1]
        v_3 = unbind_3[2]
        unbind_3 = None
        k_softmax_3 = k_3.softmax(dim=2)
        k_3 = None
        transpose_22 = k_softmax_3.transpose(-1, -2)
        k_softmax_3 = None
        factor_att_6 = transpose_22 @ v_3
        transpose_22 = None
        factor_att_7 = q_3 @ factor_att_6
        factor_att_6 = None
        q_img_3 = q_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_3 = None
        v_img_6 = v_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_3 = None
        transpose_23 = v_img_6.transpose(-1, -2)
        v_img_6 = None
        v_img_7 = transpose_23.reshape(1, 152, 28, 28)
        transpose_23 = None
        split_3 = torch.functional.split(v_img_7, [38, 57, 57], dim=1)
        v_img_7 = None
        getitem_38 = split_3[0]
        getitem_39 = split_3[1]
        getitem_40 = split_3[2]
        split_3 = None
        conv2d_15 = torch.conv2d(
            getitem_38,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_38 = None
        conv2d_16 = torch.conv2d(
            getitem_39,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_39 = None
        conv2d_17 = torch.conv2d(
            getitem_40,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_40 = None
        conv_v_img_6 = torch.cat([conv2d_15, conv2d_16, conv2d_17], dim=1)
        conv2d_15 = conv2d_16 = conv2d_17 = None
        reshape_15 = conv_v_img_6.reshape(1, 8, 19, 784)
        conv_v_img_6 = None
        conv_v_img_7 = reshape_15.transpose(-1, -2)
        reshape_15 = None
        EV_hat_6 = q_img_3 * conv_v_img_7
        q_img_3 = conv_v_img_7 = None
        EV_hat_7 = torch._C._nn.pad(EV_hat_6, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_6 = None
        mul_7 = 0.22941573387056177 * factor_att_7
        factor_att_7 = None
        x_60 = mul_7 + EV_hat_7
        mul_7 = EV_hat_7 = None
        transpose_25 = x_60.transpose(1, 2)
        x_60 = None
        x_61 = transpose_25.reshape(1, 785, 152)
        transpose_25 = None
        x_62 = torch._C._nn.linear(
            x_61,
            l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_61 = l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_63 = torch.nn.functional.dropout(x_62, 0.0, False, False)
        x_62 = None
        x_64 = x_58 + x_63
        x_58 = x_63 = None
        x_65 = torch.nn.functional.layer_norm(
            x_64,
            (152,),
            l_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_66 = torch._C._nn.linear(
            x_65,
            l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_65 = l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_67 = torch._C._nn.gelu(x_66, approximate="none")
        x_66 = None
        x_68 = torch.nn.functional.dropout(x_67, 0.0, False, False)
        x_67 = None
        x_69 = torch._C._nn.linear(
            x_68,
            l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_68 = l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_70 = torch.nn.functional.dropout(x_69, 0.0, False, False)
        x_69 = None
        x_71 = x_64 + x_70
        x_64 = x_70 = None
        getitem_41 = x_71[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        reshape_17 = getitem_41.reshape(1, 28, 28, -1)
        getitem_41 = None
        permute_5 = reshape_17.permute(0, 3, 1, 2)
        reshape_17 = None
        x2_nocls = permute_5.contiguous()
        permute_5 = None
        x_72 = torch.conv2d(
            x2_nocls,
            l_self_modules_patch_embed3_modules_proj_parameters_weight_,
            l_self_modules_patch_embed3_modules_proj_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x2_nocls = (
            l_self_modules_patch_embed3_modules_proj_parameters_weight_
        ) = l_self_modules_patch_embed3_modules_proj_parameters_bias_ = None
        flatten_6 = x_72.flatten(2)
        x_72 = None
        x_73 = flatten_6.transpose(1, 2)
        flatten_6 = None
        x_74 = torch.nn.functional.layer_norm(
            x_73,
            (152,),
            l_self_modules_patch_embed3_modules_norm_parameters_weight_,
            l_self_modules_patch_embed3_modules_norm_parameters_bias_,
            1e-05,
        )
        x_73 = (
            l_self_modules_patch_embed3_modules_norm_parameters_weight_
        ) = l_self_modules_patch_embed3_modules_norm_parameters_bias_ = None
        cls_tokens_2 = l_self_parameters_cls_token3_.expand(1, -1, -1)
        l_self_parameters_cls_token3_ = None
        x_75 = torch.cat((cls_tokens_2, x_74), dim=1)
        cls_tokens_2 = x_74 = None
        cls_token_4 = x_75[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_4 = x_75[(slice(None, None, None), slice(1, None, None))]
        x_75 = None
        transpose_27 = img_tokens_4.transpose(1, 2)
        img_tokens_4 = None
        feat_4 = transpose_27.view(1, 152, 14, 14)
        transpose_27 = None
        conv2d_19 = torch.conv2d(
            feat_4,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_76 = conv2d_19 + feat_4
        conv2d_19 = feat_4 = None
        flatten_7 = x_76.flatten(2)
        x_76 = None
        x_77 = flatten_7.transpose(1, 2)
        flatten_7 = None
        x_78 = torch.cat((cls_token_4, x_77), dim=1)
        cls_token_4 = x_77 = None
        x_79 = torch.nn.functional.layer_norm(
            x_78,
            (152,),
            l_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear_16 = torch._C._nn.linear(
            x_79,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_79 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_18 = linear_16.reshape(1, 197, 3, 8, 19)
        linear_16 = None
        qkv_4 = reshape_18.permute(2, 0, 3, 1, 4)
        reshape_18 = None
        unbind_4 = qkv_4.unbind(0)
        qkv_4 = None
        q_4 = unbind_4[0]
        k_4 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        k_softmax_4 = k_4.softmax(dim=2)
        k_4 = None
        transpose_29 = k_softmax_4.transpose(-1, -2)
        k_softmax_4 = None
        factor_att_8 = transpose_29 @ v_4
        transpose_29 = None
        factor_att_9 = q_4 @ factor_att_8
        factor_att_8 = None
        q_img_4 = q_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_4 = None
        v_img_8 = v_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_4 = None
        transpose_30 = v_img_8.transpose(-1, -2)
        v_img_8 = None
        v_img_9 = transpose_30.reshape(1, 152, 14, 14)
        transpose_30 = None
        split_4 = torch.functional.split(v_img_9, [38, 57, 57], dim=1)
        v_img_9 = None
        getitem_49 = split_4[0]
        getitem_50 = split_4[1]
        getitem_51 = split_4[2]
        split_4 = None
        conv2d_20 = torch.conv2d(
            getitem_49,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_49 = None
        conv2d_21 = torch.conv2d(
            getitem_50,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_50 = None
        conv2d_22 = torch.conv2d(
            getitem_51,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_51 = None
        conv_v_img_8 = torch.cat([conv2d_20, conv2d_21, conv2d_22], dim=1)
        conv2d_20 = conv2d_21 = conv2d_22 = None
        reshape_20 = conv_v_img_8.reshape(1, 8, 19, 196)
        conv_v_img_8 = None
        conv_v_img_9 = reshape_20.transpose(-1, -2)
        reshape_20 = None
        EV_hat_8 = q_img_4 * conv_v_img_9
        q_img_4 = conv_v_img_9 = None
        EV_hat_9 = torch._C._nn.pad(EV_hat_8, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_8 = None
        mul_9 = 0.22941573387056177 * factor_att_9
        factor_att_9 = None
        x_80 = mul_9 + EV_hat_9
        mul_9 = EV_hat_9 = None
        transpose_32 = x_80.transpose(1, 2)
        x_80 = None
        x_81 = transpose_32.reshape(1, 197, 152)
        transpose_32 = None
        x_82 = torch._C._nn.linear(
            x_81,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_81 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_83 = torch.nn.functional.dropout(x_82, 0.0, False, False)
        x_82 = None
        x_84 = x_78 + x_83
        x_78 = x_83 = None
        x_85 = torch.nn.functional.layer_norm(
            x_84,
            (152,),
            l_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_86 = torch._C._nn.linear(
            x_85,
            l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_85 = l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_87 = torch._C._nn.gelu(x_86, approximate="none")
        x_86 = None
        x_88 = torch.nn.functional.dropout(x_87, 0.0, False, False)
        x_87 = None
        x_89 = torch._C._nn.linear(
            x_88,
            l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_88 = l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_90 = torch.nn.functional.dropout(x_89, 0.0, False, False)
        x_89 = None
        x_91 = x_84 + x_90
        x_84 = x_90 = None
        cls_token_5 = x_91[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_5 = x_91[(slice(None, None, None), slice(1, None, None))]
        x_91 = None
        transpose_33 = img_tokens_5.transpose(1, 2)
        img_tokens_5 = None
        feat_5 = transpose_33.view(1, 152, 14, 14)
        transpose_33 = None
        conv2d_23 = torch.conv2d(
            feat_5,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_92 = conv2d_23 + feat_5
        conv2d_23 = feat_5 = None
        flatten_8 = x_92.flatten(2)
        x_92 = None
        x_93 = flatten_8.transpose(1, 2)
        flatten_8 = None
        x_94 = torch.cat((cls_token_5, x_93), dim=1)
        cls_token_5 = x_93 = None
        x_95 = torch.nn.functional.layer_norm(
            x_94,
            (152,),
            l_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_bias_
        ) = None
        linear_20 = torch._C._nn.linear(
            x_95,
            l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_95 = l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_22 = linear_20.reshape(1, 197, 3, 8, 19)
        linear_20 = None
        qkv_5 = reshape_22.permute(2, 0, 3, 1, 4)
        reshape_22 = None
        unbind_5 = qkv_5.unbind(0)
        qkv_5 = None
        q_5 = unbind_5[0]
        k_5 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        k_softmax_5 = k_5.softmax(dim=2)
        k_5 = None
        transpose_35 = k_softmax_5.transpose(-1, -2)
        k_softmax_5 = None
        factor_att_10 = transpose_35 @ v_5
        transpose_35 = None
        factor_att_11 = q_5 @ factor_att_10
        factor_att_10 = None
        q_img_5 = q_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_5 = None
        v_img_10 = v_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_5 = None
        transpose_36 = v_img_10.transpose(-1, -2)
        v_img_10 = None
        v_img_11 = transpose_36.reshape(1, 152, 14, 14)
        transpose_36 = None
        split_5 = torch.functional.split(v_img_11, [38, 57, 57], dim=1)
        v_img_11 = None
        getitem_59 = split_5[0]
        getitem_60 = split_5[1]
        getitem_61 = split_5[2]
        split_5 = None
        conv2d_24 = torch.conv2d(
            getitem_59,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_59 = None
        conv2d_25 = torch.conv2d(
            getitem_60,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_60 = None
        conv2d_26 = torch.conv2d(
            getitem_61,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_61 = None
        conv_v_img_10 = torch.cat([conv2d_24, conv2d_25, conv2d_26], dim=1)
        conv2d_24 = conv2d_25 = conv2d_26 = None
        reshape_24 = conv_v_img_10.reshape(1, 8, 19, 196)
        conv_v_img_10 = None
        conv_v_img_11 = reshape_24.transpose(-1, -2)
        reshape_24 = None
        EV_hat_10 = q_img_5 * conv_v_img_11
        q_img_5 = conv_v_img_11 = None
        EV_hat_11 = torch._C._nn.pad(EV_hat_10, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_10 = None
        mul_11 = 0.22941573387056177 * factor_att_11
        factor_att_11 = None
        x_96 = mul_11 + EV_hat_11
        mul_11 = EV_hat_11 = None
        transpose_38 = x_96.transpose(1, 2)
        x_96 = None
        x_97 = transpose_38.reshape(1, 197, 152)
        transpose_38 = None
        x_98 = torch._C._nn.linear(
            x_97,
            l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_97 = l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_99 = torch.nn.functional.dropout(x_98, 0.0, False, False)
        x_98 = None
        x_100 = x_94 + x_99
        x_94 = x_99 = None
        x_101 = torch.nn.functional.layer_norm(
            x_100,
            (152,),
            l_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_102 = torch._C._nn.linear(
            x_101,
            l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_101 = l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_103 = torch._C._nn.gelu(x_102, approximate="none")
        x_102 = None
        x_104 = torch.nn.functional.dropout(x_103, 0.0, False, False)
        x_103 = None
        x_105 = torch._C._nn.linear(
            x_104,
            l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_104 = l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        x_107 = x_100 + x_106
        x_100 = x_106 = None
        getitem_62 = x_107[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        reshape_26 = getitem_62.reshape(1, 14, 14, -1)
        getitem_62 = None
        permute_8 = reshape_26.permute(0, 3, 1, 2)
        reshape_26 = None
        x3_nocls = permute_8.contiguous()
        permute_8 = None
        x_108 = torch.conv2d(
            x3_nocls,
            l_self_modules_patch_embed4_modules_proj_parameters_weight_,
            l_self_modules_patch_embed4_modules_proj_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x3_nocls = (
            l_self_modules_patch_embed4_modules_proj_parameters_weight_
        ) = l_self_modules_patch_embed4_modules_proj_parameters_bias_ = None
        flatten_9 = x_108.flatten(2)
        x_108 = None
        x_109 = flatten_9.transpose(1, 2)
        flatten_9 = None
        x_110 = torch.nn.functional.layer_norm(
            x_109,
            (152,),
            l_self_modules_patch_embed4_modules_norm_parameters_weight_,
            l_self_modules_patch_embed4_modules_norm_parameters_bias_,
            1e-05,
        )
        x_109 = (
            l_self_modules_patch_embed4_modules_norm_parameters_weight_
        ) = l_self_modules_patch_embed4_modules_norm_parameters_bias_ = None
        cls_tokens_3 = l_self_parameters_cls_token4_.expand(1, -1, -1)
        l_self_parameters_cls_token4_ = None
        x_111 = torch.cat((cls_tokens_3, x_110), dim=1)
        cls_tokens_3 = x_110 = None
        cls_token_6 = x_111[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_6 = x_111[(slice(None, None, None), slice(1, None, None))]
        x_111 = None
        transpose_40 = img_tokens_6.transpose(1, 2)
        img_tokens_6 = None
        feat_6 = transpose_40.view(1, 152, 7, 7)
        transpose_40 = None
        conv2d_28 = torch.conv2d(
            feat_6,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_112 = conv2d_28 + feat_6
        conv2d_28 = feat_6 = None
        flatten_10 = x_112.flatten(2)
        x_112 = None
        x_113 = flatten_10.transpose(1, 2)
        flatten_10 = None
        x_114 = torch.cat((cls_token_6, x_113), dim=1)
        cls_token_6 = x_113 = None
        x_115 = torch.nn.functional.layer_norm(
            x_114,
            (152,),
            l_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear_24 = torch._C._nn.linear(
            x_115,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_115 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_27 = linear_24.reshape(1, 50, 3, 8, 19)
        linear_24 = None
        qkv_6 = reshape_27.permute(2, 0, 3, 1, 4)
        reshape_27 = None
        unbind_6 = qkv_6.unbind(0)
        qkv_6 = None
        q_6 = unbind_6[0]
        k_6 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        k_softmax_6 = k_6.softmax(dim=2)
        k_6 = None
        transpose_42 = k_softmax_6.transpose(-1, -2)
        k_softmax_6 = None
        factor_att_12 = transpose_42 @ v_6
        transpose_42 = None
        factor_att_13 = q_6 @ factor_att_12
        factor_att_12 = None
        q_img_6 = q_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_6 = None
        v_img_12 = v_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_6 = None
        transpose_43 = v_img_12.transpose(-1, -2)
        v_img_12 = None
        v_img_13 = transpose_43.reshape(1, 152, 7, 7)
        transpose_43 = None
        split_6 = torch.functional.split(v_img_13, [38, 57, 57], dim=1)
        v_img_13 = None
        getitem_70 = split_6[0]
        getitem_71 = split_6[1]
        getitem_72 = split_6[2]
        split_6 = None
        conv2d_29 = torch.conv2d(
            getitem_70,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_70 = None
        conv2d_30 = torch.conv2d(
            getitem_71,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_71 = None
        conv2d_31 = torch.conv2d(
            getitem_72,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_72 = None
        conv_v_img_12 = torch.cat([conv2d_29, conv2d_30, conv2d_31], dim=1)
        conv2d_29 = conv2d_30 = conv2d_31 = None
        reshape_29 = conv_v_img_12.reshape(1, 8, 19, 49)
        conv_v_img_12 = None
        conv_v_img_13 = reshape_29.transpose(-1, -2)
        reshape_29 = None
        EV_hat_12 = q_img_6 * conv_v_img_13
        q_img_6 = conv_v_img_13 = None
        EV_hat_13 = torch._C._nn.pad(EV_hat_12, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_12 = None
        mul_13 = 0.22941573387056177 * factor_att_13
        factor_att_13 = None
        x_116 = mul_13 + EV_hat_13
        mul_13 = EV_hat_13 = None
        transpose_45 = x_116.transpose(1, 2)
        x_116 = None
        x_117 = transpose_45.reshape(1, 50, 152)
        transpose_45 = None
        x_118 = torch._C._nn.linear(
            x_117,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_117 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_119 = torch.nn.functional.dropout(x_118, 0.0, False, False)
        x_118 = None
        x_120 = x_114 + x_119
        x_114 = x_119 = None
        x_121 = torch.nn.functional.layer_norm(
            x_120,
            (152,),
            l_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_122 = torch._C._nn.linear(
            x_121,
            l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_121 = l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_123 = torch._C._nn.gelu(x_122, approximate="none")
        x_122 = None
        x_124 = torch.nn.functional.dropout(x_123, 0.0, False, False)
        x_123 = None
        x_125 = torch._C._nn.linear(
            x_124,
            l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_124 = l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_126 = torch.nn.functional.dropout(x_125, 0.0, False, False)
        x_125 = None
        x_127 = x_120 + x_126
        x_120 = x_126 = None
        cls_token_7 = x_127[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_7 = x_127[(slice(None, None, None), slice(1, None, None))]
        x_127 = None
        transpose_46 = img_tokens_7.transpose(1, 2)
        img_tokens_7 = None
        feat_7 = transpose_46.view(1, 152, 7, 7)
        transpose_46 = None
        conv2d_32 = torch.conv2d(
            feat_7,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_128 = conv2d_32 + feat_7
        conv2d_32 = feat_7 = None
        flatten_11 = x_128.flatten(2)
        x_128 = None
        x_129 = flatten_11.transpose(1, 2)
        flatten_11 = None
        x_130 = torch.cat((cls_token_7, x_129), dim=1)
        cls_token_7 = x_129 = None
        x_131 = torch.nn.functional.layer_norm(
            x_130,
            (152,),
            l_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_bias_
        ) = None
        linear_28 = torch._C._nn.linear(
            x_131,
            l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_131 = l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_31 = linear_28.reshape(1, 50, 3, 8, 19)
        linear_28 = None
        qkv_7 = reshape_31.permute(2, 0, 3, 1, 4)
        reshape_31 = None
        unbind_7 = qkv_7.unbind(0)
        qkv_7 = None
        q_7 = unbind_7[0]
        k_7 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        k_softmax_7 = k_7.softmax(dim=2)
        k_7 = None
        transpose_48 = k_softmax_7.transpose(-1, -2)
        k_softmax_7 = None
        factor_att_14 = transpose_48 @ v_7
        transpose_48 = None
        factor_att_15 = q_7 @ factor_att_14
        factor_att_14 = None
        q_img_7 = q_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_7 = None
        v_img_14 = v_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_7 = None
        transpose_49 = v_img_14.transpose(-1, -2)
        v_img_14 = None
        v_img_15 = transpose_49.reshape(1, 152, 7, 7)
        transpose_49 = None
        split_7 = torch.functional.split(v_img_15, [38, 57, 57], dim=1)
        v_img_15 = None
        getitem_80 = split_7[0]
        getitem_81 = split_7[1]
        getitem_82 = split_7[2]
        split_7 = None
        conv2d_33 = torch.conv2d(
            getitem_80,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_80 = None
        conv2d_34 = torch.conv2d(
            getitem_81,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_81 = None
        conv2d_35 = torch.conv2d(
            getitem_82,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_82 = None
        conv_v_img_14 = torch.cat([conv2d_33, conv2d_34, conv2d_35], dim=1)
        conv2d_33 = conv2d_34 = conv2d_35 = None
        reshape_33 = conv_v_img_14.reshape(1, 8, 19, 49)
        conv_v_img_14 = None
        conv_v_img_15 = reshape_33.transpose(-1, -2)
        reshape_33 = None
        EV_hat_14 = q_img_7 * conv_v_img_15
        q_img_7 = conv_v_img_15 = None
        EV_hat_15 = torch._C._nn.pad(EV_hat_14, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_14 = None
        mul_15 = 0.22941573387056177 * factor_att_15
        factor_att_15 = None
        x_132 = mul_15 + EV_hat_15
        mul_15 = EV_hat_15 = None
        transpose_51 = x_132.transpose(1, 2)
        x_132 = None
        x_133 = transpose_51.reshape(1, 50, 152)
        transpose_51 = None
        x_134 = torch._C._nn.linear(
            x_133,
            l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_133 = l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_135 = torch.nn.functional.dropout(x_134, 0.0, False, False)
        x_134 = None
        x_136 = x_130 + x_135
        x_130 = x_135 = None
        x_137 = torch.nn.functional.layer_norm(
            x_136,
            (152,),
            l_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_138 = torch._C._nn.linear(
            x_137,
            l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_137 = l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_139 = torch._C._nn.gelu(x_138, approximate="none")
        x_138 = None
        x_140 = torch.nn.functional.dropout(x_139, 0.0, False, False)
        x_139 = None
        x_141 = torch._C._nn.linear(
            x_140,
            l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_140 = l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_142 = torch.nn.functional.dropout(x_141, 0.0, False, False)
        x_141 = None
        x_143 = x_136 + x_142
        x_136 = x_142 = None
        getitem_83 = x_143[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        reshape_35 = getitem_83.reshape(1, 7, 7, -1)
        getitem_83 = None
        permute_11 = reshape_35.permute(0, 3, 1, 2)
        reshape_35 = None
        x4_nocls = permute_11.contiguous()
        permute_11 = x4_nocls = None
        cls_token_8 = x_71[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_8 = x_71[(slice(None, None, None), slice(1, None, None))]
        x_71 = None
        transpose_52 = img_tokens_8.transpose(1, 2)
        img_tokens_8 = None
        feat_8 = transpose_52.view(1, 152, 28, 28)
        transpose_52 = None
        conv2d_36 = torch.conv2d(
            feat_8,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_144 = conv2d_36 + feat_8
        conv2d_36 = feat_8 = None
        flatten_12 = x_144.flatten(2)
        x_144 = None
        x_145 = flatten_12.transpose(1, 2)
        flatten_12 = None
        x_146 = torch.cat((cls_token_8, x_145), dim=1)
        cls_token_8 = x_145 = None
        cls_token_9 = x_107[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_9 = x_107[(slice(None, None, None), slice(1, None, None))]
        x_107 = None
        transpose_54 = img_tokens_9.transpose(1, 2)
        img_tokens_9 = None
        feat_9 = transpose_54.view(1, 152, 14, 14)
        transpose_54 = None
        conv2d_37 = torch.conv2d(
            feat_9,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_147 = conv2d_37 + feat_9
        conv2d_37 = feat_9 = None
        flatten_13 = x_147.flatten(2)
        x_147 = None
        x_148 = flatten_13.transpose(1, 2)
        flatten_13 = None
        x_149 = torch.cat((cls_token_9, x_148), dim=1)
        cls_token_9 = x_148 = None
        cls_token_10 = x_143[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_10 = x_143[(slice(None, None, None), slice(1, None, None))]
        x_143 = None
        transpose_56 = img_tokens_10.transpose(1, 2)
        img_tokens_10 = None
        feat_10 = transpose_56.view(1, 152, 7, 7)
        transpose_56 = None
        conv2d_38 = torch.conv2d(
            feat_10,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_150 = conv2d_38 + feat_10
        conv2d_38 = feat_10 = None
        flatten_14 = x_150.flatten(2)
        x_150 = None
        x_151 = flatten_14.transpose(1, 2)
        flatten_14 = None
        x_152 = torch.cat((cls_token_10, x_151), dim=1)
        cls_token_10 = x_151 = None
        x_153 = torch.nn.functional.layer_norm(
            x_146,
            (152,),
            l_self_modules_parallel_blocks_modules_0_modules_norm12_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_norm12_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm12_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_0_modules_norm12_parameters_bias_
        ) = None
        x_154 = torch.nn.functional.layer_norm(
            x_149,
            (152,),
            l_self_modules_parallel_blocks_modules_0_modules_norm13_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_norm13_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm13_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_0_modules_norm13_parameters_bias_
        ) = None
        x_155 = torch.nn.functional.layer_norm(
            x_152,
            (152,),
            l_self_modules_parallel_blocks_modules_0_modules_norm14_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_norm14_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm14_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_0_modules_norm14_parameters_bias_
        ) = None
        linear_32 = torch._C._nn.linear(
            x_153,
            l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_qkv_parameters_bias_,
        )
        x_153 = l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_qkv_parameters_bias_ = (None)
        reshape_36 = linear_32.reshape(1, 785, 3, 8, 19)
        linear_32 = None
        qkv_8 = reshape_36.permute(2, 0, 3, 1, 4)
        reshape_36 = None
        unbind_8 = qkv_8.unbind(0)
        qkv_8 = None
        q_8 = unbind_8[0]
        k_8 = unbind_8[1]
        v_8 = unbind_8[2]
        unbind_8 = None
        k_softmax_8 = k_8.softmax(dim=2)
        k_8 = None
        transpose_58 = k_softmax_8.transpose(-1, -2)
        k_softmax_8 = None
        factor_att_16 = transpose_58 @ v_8
        transpose_58 = None
        factor_att_17 = q_8 @ factor_att_16
        factor_att_16 = None
        q_img_8 = q_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_8 = None
        v_img_16 = v_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_8 = None
        transpose_59 = v_img_16.transpose(-1, -2)
        v_img_16 = None
        v_img_17 = transpose_59.reshape(1, 152, 28, 28)
        transpose_59 = None
        split_8 = torch.functional.split(v_img_17, [38, 57, 57], dim=1)
        v_img_17 = None
        getitem_95 = split_8[0]
        getitem_96 = split_8[1]
        getitem_97 = split_8[2]
        split_8 = None
        conv2d_39 = torch.conv2d(
            getitem_95,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_95 = None
        conv2d_40 = torch.conv2d(
            getitem_96,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_96 = None
        conv2d_41 = torch.conv2d(
            getitem_97,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_97 = None
        conv_v_img_16 = torch.cat([conv2d_39, conv2d_40, conv2d_41], dim=1)
        conv2d_39 = conv2d_40 = conv2d_41 = None
        reshape_38 = conv_v_img_16.reshape(1, 8, 19, 784)
        conv_v_img_16 = None
        conv_v_img_17 = reshape_38.transpose(-1, -2)
        reshape_38 = None
        EV_hat_16 = q_img_8 * conv_v_img_17
        q_img_8 = conv_v_img_17 = None
        EV_hat_17 = torch._C._nn.pad(EV_hat_16, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_16 = None
        mul_17 = 0.22941573387056177 * factor_att_17
        factor_att_17 = None
        x_156 = mul_17 + EV_hat_17
        mul_17 = EV_hat_17 = None
        transpose_61 = x_156.transpose(1, 2)
        x_156 = None
        x_157 = transpose_61.reshape(1, 785, 152)
        transpose_61 = None
        x_158 = torch._C._nn.linear(
            x_157,
            l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_proj_parameters_bias_,
        )
        x_157 = l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe2_modules_proj_parameters_bias_ = (None)
        x_159 = torch.nn.functional.dropout(x_158, 0.0, False, False)
        x_158 = None
        linear_34 = torch._C._nn.linear(
            x_154,
            l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_qkv_parameters_bias_,
        )
        x_154 = l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_qkv_parameters_bias_ = (None)
        reshape_40 = linear_34.reshape(1, 197, 3, 8, 19)
        linear_34 = None
        qkv_9 = reshape_40.permute(2, 0, 3, 1, 4)
        reshape_40 = None
        unbind_9 = qkv_9.unbind(0)
        qkv_9 = None
        q_9 = unbind_9[0]
        k_9 = unbind_9[1]
        v_9 = unbind_9[2]
        unbind_9 = None
        k_softmax_9 = k_9.softmax(dim=2)
        k_9 = None
        transpose_62 = k_softmax_9.transpose(-1, -2)
        k_softmax_9 = None
        factor_att_18 = transpose_62 @ v_9
        transpose_62 = None
        factor_att_19 = q_9 @ factor_att_18
        factor_att_18 = None
        q_img_9 = q_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_9 = None
        v_img_18 = v_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_9 = None
        transpose_63 = v_img_18.transpose(-1, -2)
        v_img_18 = None
        v_img_19 = transpose_63.reshape(1, 152, 14, 14)
        transpose_63 = None
        split_9 = torch.functional.split(v_img_19, [38, 57, 57], dim=1)
        v_img_19 = None
        getitem_103 = split_9[0]
        getitem_104 = split_9[1]
        getitem_105 = split_9[2]
        split_9 = None
        conv2d_42 = torch.conv2d(
            getitem_103,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_103 = None
        conv2d_43 = torch.conv2d(
            getitem_104,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_104 = None
        conv2d_44 = torch.conv2d(
            getitem_105,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_105 = None
        conv_v_img_18 = torch.cat([conv2d_42, conv2d_43, conv2d_44], dim=1)
        conv2d_42 = conv2d_43 = conv2d_44 = None
        reshape_42 = conv_v_img_18.reshape(1, 8, 19, 196)
        conv_v_img_18 = None
        conv_v_img_19 = reshape_42.transpose(-1, -2)
        reshape_42 = None
        EV_hat_18 = q_img_9 * conv_v_img_19
        q_img_9 = conv_v_img_19 = None
        EV_hat_19 = torch._C._nn.pad(EV_hat_18, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_18 = None
        mul_19 = 0.22941573387056177 * factor_att_19
        factor_att_19 = None
        x_160 = mul_19 + EV_hat_19
        mul_19 = EV_hat_19 = None
        transpose_65 = x_160.transpose(1, 2)
        x_160 = None
        x_161 = transpose_65.reshape(1, 197, 152)
        transpose_65 = None
        x_162 = torch._C._nn.linear(
            x_161,
            l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_proj_parameters_bias_,
        )
        x_161 = l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe3_modules_proj_parameters_bias_ = (None)
        x_163 = torch.nn.functional.dropout(x_162, 0.0, False, False)
        x_162 = None
        linear_36 = torch._C._nn.linear(
            x_155,
            l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_qkv_parameters_bias_,
        )
        x_155 = l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_qkv_parameters_bias_ = (None)
        reshape_44 = linear_36.reshape(1, 50, 3, 8, 19)
        linear_36 = None
        qkv_10 = reshape_44.permute(2, 0, 3, 1, 4)
        reshape_44 = None
        unbind_10 = qkv_10.unbind(0)
        qkv_10 = None
        q_10 = unbind_10[0]
        k_10 = unbind_10[1]
        v_10 = unbind_10[2]
        unbind_10 = None
        k_softmax_10 = k_10.softmax(dim=2)
        k_10 = None
        transpose_66 = k_softmax_10.transpose(-1, -2)
        k_softmax_10 = None
        factor_att_20 = transpose_66 @ v_10
        transpose_66 = None
        factor_att_21 = q_10 @ factor_att_20
        factor_att_20 = None
        q_img_10 = q_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_10 = None
        v_img_20 = v_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_10 = None
        transpose_67 = v_img_20.transpose(-1, -2)
        v_img_20 = None
        v_img_21 = transpose_67.reshape(1, 152, 7, 7)
        transpose_67 = None
        split_10 = torch.functional.split(v_img_21, [38, 57, 57], dim=1)
        v_img_21 = None
        getitem_111 = split_10[0]
        getitem_112 = split_10[1]
        getitem_113 = split_10[2]
        split_10 = None
        conv2d_45 = torch.conv2d(
            getitem_111,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_111 = None
        conv2d_46 = torch.conv2d(
            getitem_112,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_112 = None
        conv2d_47 = torch.conv2d(
            getitem_113,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_113 = None
        conv_v_img_20 = torch.cat([conv2d_45, conv2d_46, conv2d_47], dim=1)
        conv2d_45 = conv2d_46 = conv2d_47 = None
        reshape_46 = conv_v_img_20.reshape(1, 8, 19, 49)
        conv_v_img_20 = None
        conv_v_img_21 = reshape_46.transpose(-1, -2)
        reshape_46 = None
        EV_hat_20 = q_img_10 * conv_v_img_21
        q_img_10 = conv_v_img_21 = None
        EV_hat_21 = torch._C._nn.pad(EV_hat_20, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_20 = None
        mul_21 = 0.22941573387056177 * factor_att_21
        factor_att_21 = None
        x_164 = mul_21 + EV_hat_21
        mul_21 = EV_hat_21 = None
        transpose_69 = x_164.transpose(1, 2)
        x_164 = None
        x_165 = transpose_69.reshape(1, 50, 152)
        transpose_69 = None
        x_166 = torch._C._nn.linear(
            x_165,
            l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_proj_parameters_bias_,
        )
        x_165 = l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_0_modules_factoratt_crpe4_modules_proj_parameters_bias_ = (None)
        x_167 = torch.nn.functional.dropout(x_166, 0.0, False, False)
        x_166 = None
        cls_token_11 = x_163[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_11 = x_163[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_70 = img_tokens_11.transpose(1, 2)
        img_tokens_11 = None
        img_tokens_12 = transpose_70.reshape(1, 152, 14, 14)
        transpose_70 = None
        img_tokens_13 = torch.nn.functional.interpolate(
            img_tokens_12,
            scale_factor=2.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_12 = None
        reshape_49 = img_tokens_13.reshape(1, 152, -1)
        img_tokens_13 = None
        img_tokens_14 = reshape_49.transpose(1, 2)
        reshape_49 = None
        out = torch.cat((cls_token_11, img_tokens_14), dim=1)
        cls_token_11 = img_tokens_14 = None
        cls_token_12 = x_167[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_15 = x_167[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_72 = img_tokens_15.transpose(1, 2)
        img_tokens_15 = None
        img_tokens_16 = transpose_72.reshape(1, 152, 7, 7)
        transpose_72 = None
        img_tokens_17 = torch.nn.functional.interpolate(
            img_tokens_16,
            scale_factor=2.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_16 = None
        reshape_51 = img_tokens_17.reshape(1, 152, -1)
        img_tokens_17 = None
        img_tokens_18 = reshape_51.transpose(1, 2)
        reshape_51 = None
        out_1 = torch.cat((cls_token_12, img_tokens_18), dim=1)
        cls_token_12 = img_tokens_18 = None
        cls_token_13 = x_167[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_19 = x_167[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_74 = img_tokens_19.transpose(1, 2)
        img_tokens_19 = None
        img_tokens_20 = transpose_74.reshape(1, 152, 7, 7)
        transpose_74 = None
        img_tokens_21 = torch.nn.functional.interpolate(
            img_tokens_20,
            scale_factor=4.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_20 = None
        reshape_53 = img_tokens_21.reshape(1, 152, -1)
        img_tokens_21 = None
        img_tokens_22 = reshape_53.transpose(1, 2)
        reshape_53 = None
        out_2 = torch.cat((cls_token_13, img_tokens_22), dim=1)
        cls_token_13 = img_tokens_22 = None
        cls_token_14 = x_159[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_23 = x_159[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_76 = img_tokens_23.transpose(1, 2)
        img_tokens_23 = None
        img_tokens_24 = transpose_76.reshape(1, 152, 28, 28)
        transpose_76 = None
        img_tokens_25 = torch.nn.functional.interpolate(
            img_tokens_24,
            scale_factor=0.5,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_24 = None
        reshape_55 = img_tokens_25.reshape(1, 152, -1)
        img_tokens_25 = None
        img_tokens_26 = reshape_55.transpose(1, 2)
        reshape_55 = None
        out_3 = torch.cat((cls_token_14, img_tokens_26), dim=1)
        cls_token_14 = img_tokens_26 = None
        cls_token_15 = x_163[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_27 = x_163[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_78 = img_tokens_27.transpose(1, 2)
        img_tokens_27 = None
        img_tokens_28 = transpose_78.reshape(1, 152, 14, 14)
        transpose_78 = None
        img_tokens_29 = torch.nn.functional.interpolate(
            img_tokens_28,
            scale_factor=0.5,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_28 = None
        reshape_57 = img_tokens_29.reshape(1, 152, -1)
        img_tokens_29 = None
        img_tokens_30 = reshape_57.transpose(1, 2)
        reshape_57 = None
        out_4 = torch.cat((cls_token_15, img_tokens_30), dim=1)
        cls_token_15 = img_tokens_30 = None
        cls_token_16 = x_159[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_31 = x_159[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_80 = img_tokens_31.transpose(1, 2)
        img_tokens_31 = None
        img_tokens_32 = transpose_80.reshape(1, 152, 28, 28)
        transpose_80 = None
        img_tokens_33 = torch.nn.functional.interpolate(
            img_tokens_32,
            scale_factor=0.25,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_32 = None
        reshape_59 = img_tokens_33.reshape(1, 152, -1)
        img_tokens_33 = None
        img_tokens_34 = reshape_59.transpose(1, 2)
        reshape_59 = None
        out_5 = torch.cat((cls_token_16, img_tokens_34), dim=1)
        cls_token_16 = img_tokens_34 = None
        add_38 = x_159 + out
        x_159 = out = None
        cur2 = add_38 + out_2
        add_38 = out_2 = None
        add_40 = x_163 + out_1
        x_163 = out_1 = None
        cur3 = add_40 + out_3
        add_40 = out_3 = None
        add_42 = x_167 + out_4
        x_167 = out_4 = None
        cur4 = add_42 + out_5
        add_42 = out_5 = None
        x2 = x_146 + cur2
        x_146 = cur2 = None
        x3 = x_149 + cur3
        x_149 = cur3 = None
        x4 = x_152 + cur4
        x_152 = cur4 = None
        x_168 = torch.nn.functional.layer_norm(
            x2,
            (152,),
            l_self_modules_parallel_blocks_modules_0_modules_norm22_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_norm22_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm22_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_0_modules_norm22_parameters_bias_
        ) = None
        x_169 = torch.nn.functional.layer_norm(
            x3,
            (152,),
            l_self_modules_parallel_blocks_modules_0_modules_norm23_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_norm23_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm23_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_0_modules_norm23_parameters_bias_
        ) = None
        x_170 = torch.nn.functional.layer_norm(
            x4,
            (152,),
            l_self_modules_parallel_blocks_modules_0_modules_norm24_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_norm24_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_0_modules_norm24_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_0_modules_norm24_parameters_bias_
        ) = None
        x_171 = torch._C._nn.linear(
            x_168,
            l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_168 = None
        x_172 = torch._C._nn.gelu(x_171, approximate="none")
        x_171 = None
        x_173 = torch.nn.functional.dropout(x_172, 0.0, False, False)
        x_172 = None
        x_174 = torch._C._nn.linear(
            x_173,
            l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_173 = None
        x_175 = torch.nn.functional.dropout(x_174, 0.0, False, False)
        x_174 = None
        x_176 = torch._C._nn.linear(
            x_169,
            l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_169 = None
        x_177 = torch._C._nn.gelu(x_176, approximate="none")
        x_176 = None
        x_178 = torch.nn.functional.dropout(x_177, 0.0, False, False)
        x_177 = None
        x_179 = torch._C._nn.linear(
            x_178,
            l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_178 = None
        x_180 = torch.nn.functional.dropout(x_179, 0.0, False, False)
        x_179 = None
        x_181 = torch._C._nn.linear(
            x_170,
            l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_170 = l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_weight_ = l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc1_parameters_bias_ = (None)
        x_182 = torch._C._nn.gelu(x_181, approximate="none")
        x_181 = None
        x_183 = torch.nn.functional.dropout(x_182, 0.0, False, False)
        x_182 = None
        x_184 = torch._C._nn.linear(
            x_183,
            l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_183 = l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_weight_ = l_self_modules_parallel_blocks_modules_0_modules_mlp2_modules_fc2_parameters_bias_ = (None)
        x_185 = torch.nn.functional.dropout(x_184, 0.0, False, False)
        x_184 = None
        x2_1 = x2 + x_175
        x2 = x_175 = None
        x3_1 = x3 + x_180
        x3 = x_180 = None
        x4_1 = x4 + x_185
        x4 = x_185 = None
        cls_token_17 = x2_1[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_35 = x2_1[(slice(None, None, None), slice(1, None, None))]
        x2_1 = None
        transpose_82 = img_tokens_35.transpose(1, 2)
        img_tokens_35 = None
        feat_11 = transpose_82.view(1, 152, 28, 28)
        transpose_82 = None
        conv2d_48 = torch.conv2d(
            feat_11,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_186 = conv2d_48 + feat_11
        conv2d_48 = feat_11 = None
        flatten_15 = x_186.flatten(2)
        x_186 = None
        x_187 = flatten_15.transpose(1, 2)
        flatten_15 = None
        x_188 = torch.cat((cls_token_17, x_187), dim=1)
        cls_token_17 = x_187 = None
        cls_token_18 = x3_1[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_36 = x3_1[(slice(None, None, None), slice(1, None, None))]
        x3_1 = None
        transpose_84 = img_tokens_36.transpose(1, 2)
        img_tokens_36 = None
        feat_12 = transpose_84.view(1, 152, 14, 14)
        transpose_84 = None
        conv2d_49 = torch.conv2d(
            feat_12,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_189 = conv2d_49 + feat_12
        conv2d_49 = feat_12 = None
        flatten_16 = x_189.flatten(2)
        x_189 = None
        x_190 = flatten_16.transpose(1, 2)
        flatten_16 = None
        x_191 = torch.cat((cls_token_18, x_190), dim=1)
        cls_token_18 = x_190 = None
        cls_token_19 = x4_1[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_37 = x4_1[(slice(None, None, None), slice(1, None, None))]
        x4_1 = None
        transpose_86 = img_tokens_37.transpose(1, 2)
        img_tokens_37 = None
        feat_13 = transpose_86.view(1, 152, 7, 7)
        transpose_86 = None
        conv2d_50 = torch.conv2d(
            feat_13,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_192 = conv2d_50 + feat_13
        conv2d_50 = feat_13 = None
        flatten_17 = x_192.flatten(2)
        x_192 = None
        x_193 = flatten_17.transpose(1, 2)
        flatten_17 = None
        x_194 = torch.cat((cls_token_19, x_193), dim=1)
        cls_token_19 = x_193 = None
        x_195 = torch.nn.functional.layer_norm(
            x_188,
            (152,),
            l_self_modules_parallel_blocks_modules_1_modules_norm12_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_norm12_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm12_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_1_modules_norm12_parameters_bias_
        ) = None
        x_196 = torch.nn.functional.layer_norm(
            x_191,
            (152,),
            l_self_modules_parallel_blocks_modules_1_modules_norm13_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_norm13_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm13_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_1_modules_norm13_parameters_bias_
        ) = None
        x_197 = torch.nn.functional.layer_norm(
            x_194,
            (152,),
            l_self_modules_parallel_blocks_modules_1_modules_norm14_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_norm14_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm14_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_1_modules_norm14_parameters_bias_
        ) = None
        linear_44 = torch._C._nn.linear(
            x_195,
            l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_qkv_parameters_bias_,
        )
        x_195 = l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_qkv_parameters_bias_ = (None)
        reshape_60 = linear_44.reshape(1, 785, 3, 8, 19)
        linear_44 = None
        qkv_11 = reshape_60.permute(2, 0, 3, 1, 4)
        reshape_60 = None
        unbind_11 = qkv_11.unbind(0)
        qkv_11 = None
        q_11 = unbind_11[0]
        k_11 = unbind_11[1]
        v_11 = unbind_11[2]
        unbind_11 = None
        k_softmax_11 = k_11.softmax(dim=2)
        k_11 = None
        transpose_88 = k_softmax_11.transpose(-1, -2)
        k_softmax_11 = None
        factor_att_22 = transpose_88 @ v_11
        transpose_88 = None
        factor_att_23 = q_11 @ factor_att_22
        factor_att_22 = None
        q_img_11 = q_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_11 = None
        v_img_22 = v_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_11 = None
        transpose_89 = v_img_22.transpose(-1, -2)
        v_img_22 = None
        v_img_23 = transpose_89.reshape(1, 152, 28, 28)
        transpose_89 = None
        split_11 = torch.functional.split(v_img_23, [38, 57, 57], dim=1)
        v_img_23 = None
        getitem_137 = split_11[0]
        getitem_138 = split_11[1]
        getitem_139 = split_11[2]
        split_11 = None
        conv2d_51 = torch.conv2d(
            getitem_137,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_137 = None
        conv2d_52 = torch.conv2d(
            getitem_138,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_138 = None
        conv2d_53 = torch.conv2d(
            getitem_139,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_139 = None
        conv_v_img_22 = torch.cat([conv2d_51, conv2d_52, conv2d_53], dim=1)
        conv2d_51 = conv2d_52 = conv2d_53 = None
        reshape_62 = conv_v_img_22.reshape(1, 8, 19, 784)
        conv_v_img_22 = None
        conv_v_img_23 = reshape_62.transpose(-1, -2)
        reshape_62 = None
        EV_hat_22 = q_img_11 * conv_v_img_23
        q_img_11 = conv_v_img_23 = None
        EV_hat_23 = torch._C._nn.pad(EV_hat_22, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_22 = None
        mul_23 = 0.22941573387056177 * factor_att_23
        factor_att_23 = None
        x_198 = mul_23 + EV_hat_23
        mul_23 = EV_hat_23 = None
        transpose_91 = x_198.transpose(1, 2)
        x_198 = None
        x_199 = transpose_91.reshape(1, 785, 152)
        transpose_91 = None
        x_200 = torch._C._nn.linear(
            x_199,
            l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_proj_parameters_bias_,
        )
        x_199 = l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe2_modules_proj_parameters_bias_ = (None)
        x_201 = torch.nn.functional.dropout(x_200, 0.0, False, False)
        x_200 = None
        linear_46 = torch._C._nn.linear(
            x_196,
            l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_qkv_parameters_bias_,
        )
        x_196 = l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_qkv_parameters_bias_ = (None)
        reshape_64 = linear_46.reshape(1, 197, 3, 8, 19)
        linear_46 = None
        qkv_12 = reshape_64.permute(2, 0, 3, 1, 4)
        reshape_64 = None
        unbind_12 = qkv_12.unbind(0)
        qkv_12 = None
        q_12 = unbind_12[0]
        k_12 = unbind_12[1]
        v_12 = unbind_12[2]
        unbind_12 = None
        k_softmax_12 = k_12.softmax(dim=2)
        k_12 = None
        transpose_92 = k_softmax_12.transpose(-1, -2)
        k_softmax_12 = None
        factor_att_24 = transpose_92 @ v_12
        transpose_92 = None
        factor_att_25 = q_12 @ factor_att_24
        factor_att_24 = None
        q_img_12 = q_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_12 = None
        v_img_24 = v_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_12 = None
        transpose_93 = v_img_24.transpose(-1, -2)
        v_img_24 = None
        v_img_25 = transpose_93.reshape(1, 152, 14, 14)
        transpose_93 = None
        split_12 = torch.functional.split(v_img_25, [38, 57, 57], dim=1)
        v_img_25 = None
        getitem_145 = split_12[0]
        getitem_146 = split_12[1]
        getitem_147 = split_12[2]
        split_12 = None
        conv2d_54 = torch.conv2d(
            getitem_145,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_145 = None
        conv2d_55 = torch.conv2d(
            getitem_146,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_146 = None
        conv2d_56 = torch.conv2d(
            getitem_147,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_147 = None
        conv_v_img_24 = torch.cat([conv2d_54, conv2d_55, conv2d_56], dim=1)
        conv2d_54 = conv2d_55 = conv2d_56 = None
        reshape_66 = conv_v_img_24.reshape(1, 8, 19, 196)
        conv_v_img_24 = None
        conv_v_img_25 = reshape_66.transpose(-1, -2)
        reshape_66 = None
        EV_hat_24 = q_img_12 * conv_v_img_25
        q_img_12 = conv_v_img_25 = None
        EV_hat_25 = torch._C._nn.pad(EV_hat_24, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_24 = None
        mul_25 = 0.22941573387056177 * factor_att_25
        factor_att_25 = None
        x_202 = mul_25 + EV_hat_25
        mul_25 = EV_hat_25 = None
        transpose_95 = x_202.transpose(1, 2)
        x_202 = None
        x_203 = transpose_95.reshape(1, 197, 152)
        transpose_95 = None
        x_204 = torch._C._nn.linear(
            x_203,
            l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_proj_parameters_bias_,
        )
        x_203 = l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe3_modules_proj_parameters_bias_ = (None)
        x_205 = torch.nn.functional.dropout(x_204, 0.0, False, False)
        x_204 = None
        linear_48 = torch._C._nn.linear(
            x_197,
            l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_qkv_parameters_bias_,
        )
        x_197 = l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_qkv_parameters_bias_ = (None)
        reshape_68 = linear_48.reshape(1, 50, 3, 8, 19)
        linear_48 = None
        qkv_13 = reshape_68.permute(2, 0, 3, 1, 4)
        reshape_68 = None
        unbind_13 = qkv_13.unbind(0)
        qkv_13 = None
        q_13 = unbind_13[0]
        k_13 = unbind_13[1]
        v_13 = unbind_13[2]
        unbind_13 = None
        k_softmax_13 = k_13.softmax(dim=2)
        k_13 = None
        transpose_96 = k_softmax_13.transpose(-1, -2)
        k_softmax_13 = None
        factor_att_26 = transpose_96 @ v_13
        transpose_96 = None
        factor_att_27 = q_13 @ factor_att_26
        factor_att_26 = None
        q_img_13 = q_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_13 = None
        v_img_26 = v_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_13 = None
        transpose_97 = v_img_26.transpose(-1, -2)
        v_img_26 = None
        v_img_27 = transpose_97.reshape(1, 152, 7, 7)
        transpose_97 = None
        split_13 = torch.functional.split(v_img_27, [38, 57, 57], dim=1)
        v_img_27 = None
        getitem_153 = split_13[0]
        getitem_154 = split_13[1]
        getitem_155 = split_13[2]
        split_13 = None
        conv2d_57 = torch.conv2d(
            getitem_153,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_153 = None
        conv2d_58 = torch.conv2d(
            getitem_154,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_154 = None
        conv2d_59 = torch.conv2d(
            getitem_155,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_155 = None
        conv_v_img_26 = torch.cat([conv2d_57, conv2d_58, conv2d_59], dim=1)
        conv2d_57 = conv2d_58 = conv2d_59 = None
        reshape_70 = conv_v_img_26.reshape(1, 8, 19, 49)
        conv_v_img_26 = None
        conv_v_img_27 = reshape_70.transpose(-1, -2)
        reshape_70 = None
        EV_hat_26 = q_img_13 * conv_v_img_27
        q_img_13 = conv_v_img_27 = None
        EV_hat_27 = torch._C._nn.pad(EV_hat_26, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_26 = None
        mul_27 = 0.22941573387056177 * factor_att_27
        factor_att_27 = None
        x_206 = mul_27 + EV_hat_27
        mul_27 = EV_hat_27 = None
        transpose_99 = x_206.transpose(1, 2)
        x_206 = None
        x_207 = transpose_99.reshape(1, 50, 152)
        transpose_99 = None
        x_208 = torch._C._nn.linear(
            x_207,
            l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_proj_parameters_bias_,
        )
        x_207 = l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_1_modules_factoratt_crpe4_modules_proj_parameters_bias_ = (None)
        x_209 = torch.nn.functional.dropout(x_208, 0.0, False, False)
        x_208 = None
        cls_token_20 = x_205[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_38 = x_205[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_100 = img_tokens_38.transpose(1, 2)
        img_tokens_38 = None
        img_tokens_39 = transpose_100.reshape(1, 152, 14, 14)
        transpose_100 = None
        img_tokens_40 = torch.nn.functional.interpolate(
            img_tokens_39,
            scale_factor=2.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_39 = None
        reshape_73 = img_tokens_40.reshape(1, 152, -1)
        img_tokens_40 = None
        img_tokens_41 = reshape_73.transpose(1, 2)
        reshape_73 = None
        out_6 = torch.cat((cls_token_20, img_tokens_41), dim=1)
        cls_token_20 = img_tokens_41 = None
        cls_token_21 = x_209[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_42 = x_209[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_102 = img_tokens_42.transpose(1, 2)
        img_tokens_42 = None
        img_tokens_43 = transpose_102.reshape(1, 152, 7, 7)
        transpose_102 = None
        img_tokens_44 = torch.nn.functional.interpolate(
            img_tokens_43,
            scale_factor=2.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_43 = None
        reshape_75 = img_tokens_44.reshape(1, 152, -1)
        img_tokens_44 = None
        img_tokens_45 = reshape_75.transpose(1, 2)
        reshape_75 = None
        out_7 = torch.cat((cls_token_21, img_tokens_45), dim=1)
        cls_token_21 = img_tokens_45 = None
        cls_token_22 = x_209[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_46 = x_209[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_104 = img_tokens_46.transpose(1, 2)
        img_tokens_46 = None
        img_tokens_47 = transpose_104.reshape(1, 152, 7, 7)
        transpose_104 = None
        img_tokens_48 = torch.nn.functional.interpolate(
            img_tokens_47,
            scale_factor=4.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_47 = None
        reshape_77 = img_tokens_48.reshape(1, 152, -1)
        img_tokens_48 = None
        img_tokens_49 = reshape_77.transpose(1, 2)
        reshape_77 = None
        out_8 = torch.cat((cls_token_22, img_tokens_49), dim=1)
        cls_token_22 = img_tokens_49 = None
        cls_token_23 = x_201[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_50 = x_201[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_106 = img_tokens_50.transpose(1, 2)
        img_tokens_50 = None
        img_tokens_51 = transpose_106.reshape(1, 152, 28, 28)
        transpose_106 = None
        img_tokens_52 = torch.nn.functional.interpolate(
            img_tokens_51,
            scale_factor=0.5,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_51 = None
        reshape_79 = img_tokens_52.reshape(1, 152, -1)
        img_tokens_52 = None
        img_tokens_53 = reshape_79.transpose(1, 2)
        reshape_79 = None
        out_9 = torch.cat((cls_token_23, img_tokens_53), dim=1)
        cls_token_23 = img_tokens_53 = None
        cls_token_24 = x_205[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_54 = x_205[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_108 = img_tokens_54.transpose(1, 2)
        img_tokens_54 = None
        img_tokens_55 = transpose_108.reshape(1, 152, 14, 14)
        transpose_108 = None
        img_tokens_56 = torch.nn.functional.interpolate(
            img_tokens_55,
            scale_factor=0.5,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_55 = None
        reshape_81 = img_tokens_56.reshape(1, 152, -1)
        img_tokens_56 = None
        img_tokens_57 = reshape_81.transpose(1, 2)
        reshape_81 = None
        out_10 = torch.cat((cls_token_24, img_tokens_57), dim=1)
        cls_token_24 = img_tokens_57 = None
        cls_token_25 = x_201[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_58 = x_201[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_110 = img_tokens_58.transpose(1, 2)
        img_tokens_58 = None
        img_tokens_59 = transpose_110.reshape(1, 152, 28, 28)
        transpose_110 = None
        img_tokens_60 = torch.nn.functional.interpolate(
            img_tokens_59,
            scale_factor=0.25,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_59 = None
        reshape_83 = img_tokens_60.reshape(1, 152, -1)
        img_tokens_60 = None
        img_tokens_61 = reshape_83.transpose(1, 2)
        reshape_83 = None
        out_11 = torch.cat((cls_token_25, img_tokens_61), dim=1)
        cls_token_25 = img_tokens_61 = None
        add_56 = x_201 + out_6
        x_201 = out_6 = None
        cur2_1 = add_56 + out_8
        add_56 = out_8 = None
        add_58 = x_205 + out_7
        x_205 = out_7 = None
        cur3_1 = add_58 + out_9
        add_58 = out_9 = None
        add_60 = x_209 + out_10
        x_209 = out_10 = None
        cur4_1 = add_60 + out_11
        add_60 = out_11 = None
        x2_2 = x_188 + cur2_1
        x_188 = cur2_1 = None
        x3_2 = x_191 + cur3_1
        x_191 = cur3_1 = None
        x4_2 = x_194 + cur4_1
        x_194 = cur4_1 = None
        x_210 = torch.nn.functional.layer_norm(
            x2_2,
            (152,),
            l_self_modules_parallel_blocks_modules_1_modules_norm22_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_norm22_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm22_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_1_modules_norm22_parameters_bias_
        ) = None
        x_211 = torch.nn.functional.layer_norm(
            x3_2,
            (152,),
            l_self_modules_parallel_blocks_modules_1_modules_norm23_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_norm23_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm23_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_1_modules_norm23_parameters_bias_
        ) = None
        x_212 = torch.nn.functional.layer_norm(
            x4_2,
            (152,),
            l_self_modules_parallel_blocks_modules_1_modules_norm24_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_norm24_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_1_modules_norm24_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_1_modules_norm24_parameters_bias_
        ) = None
        x_213 = torch._C._nn.linear(
            x_210,
            l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_210 = None
        x_214 = torch._C._nn.gelu(x_213, approximate="none")
        x_213 = None
        x_215 = torch.nn.functional.dropout(x_214, 0.0, False, False)
        x_214 = None
        x_216 = torch._C._nn.linear(
            x_215,
            l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_215 = None
        x_217 = torch.nn.functional.dropout(x_216, 0.0, False, False)
        x_216 = None
        x_218 = torch._C._nn.linear(
            x_211,
            l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_211 = None
        x_219 = torch._C._nn.gelu(x_218, approximate="none")
        x_218 = None
        x_220 = torch.nn.functional.dropout(x_219, 0.0, False, False)
        x_219 = None
        x_221 = torch._C._nn.linear(
            x_220,
            l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_220 = None
        x_222 = torch.nn.functional.dropout(x_221, 0.0, False, False)
        x_221 = None
        x_223 = torch._C._nn.linear(
            x_212,
            l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_212 = l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_weight_ = l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc1_parameters_bias_ = (None)
        x_224 = torch._C._nn.gelu(x_223, approximate="none")
        x_223 = None
        x_225 = torch.nn.functional.dropout(x_224, 0.0, False, False)
        x_224 = None
        x_226 = torch._C._nn.linear(
            x_225,
            l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_225 = l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_weight_ = l_self_modules_parallel_blocks_modules_1_modules_mlp2_modules_fc2_parameters_bias_ = (None)
        x_227 = torch.nn.functional.dropout(x_226, 0.0, False, False)
        x_226 = None
        x2_3 = x2_2 + x_217
        x2_2 = x_217 = None
        x3_3 = x3_2 + x_222
        x3_2 = x_222 = None
        x4_3 = x4_2 + x_227
        x4_2 = x_227 = None
        cls_token_26 = x2_3[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_62 = x2_3[(slice(None, None, None), slice(1, None, None))]
        x2_3 = None
        transpose_112 = img_tokens_62.transpose(1, 2)
        img_tokens_62 = None
        feat_14 = transpose_112.view(1, 152, 28, 28)
        transpose_112 = None
        conv2d_60 = torch.conv2d(
            feat_14,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_228 = conv2d_60 + feat_14
        conv2d_60 = feat_14 = None
        flatten_18 = x_228.flatten(2)
        x_228 = None
        x_229 = flatten_18.transpose(1, 2)
        flatten_18 = None
        x_230 = torch.cat((cls_token_26, x_229), dim=1)
        cls_token_26 = x_229 = None
        cls_token_27 = x3_3[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_63 = x3_3[(slice(None, None, None), slice(1, None, None))]
        x3_3 = None
        transpose_114 = img_tokens_63.transpose(1, 2)
        img_tokens_63 = None
        feat_15 = transpose_114.view(1, 152, 14, 14)
        transpose_114 = None
        conv2d_61 = torch.conv2d(
            feat_15,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_231 = conv2d_61 + feat_15
        conv2d_61 = feat_15 = None
        flatten_19 = x_231.flatten(2)
        x_231 = None
        x_232 = flatten_19.transpose(1, 2)
        flatten_19 = None
        x_233 = torch.cat((cls_token_27, x_232), dim=1)
        cls_token_27 = x_232 = None
        cls_token_28 = x4_3[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_64 = x4_3[(slice(None, None, None), slice(1, None, None))]
        x4_3 = None
        transpose_116 = img_tokens_64.transpose(1, 2)
        img_tokens_64 = None
        feat_16 = transpose_116.view(1, 152, 7, 7)
        transpose_116 = None
        conv2d_62 = torch.conv2d(
            feat_16,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_234 = conv2d_62 + feat_16
        conv2d_62 = feat_16 = None
        flatten_20 = x_234.flatten(2)
        x_234 = None
        x_235 = flatten_20.transpose(1, 2)
        flatten_20 = None
        x_236 = torch.cat((cls_token_28, x_235), dim=1)
        cls_token_28 = x_235 = None
        x_237 = torch.nn.functional.layer_norm(
            x_230,
            (152,),
            l_self_modules_parallel_blocks_modules_2_modules_norm12_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_norm12_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm12_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_2_modules_norm12_parameters_bias_
        ) = None
        x_238 = torch.nn.functional.layer_norm(
            x_233,
            (152,),
            l_self_modules_parallel_blocks_modules_2_modules_norm13_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_norm13_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm13_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_2_modules_norm13_parameters_bias_
        ) = None
        x_239 = torch.nn.functional.layer_norm(
            x_236,
            (152,),
            l_self_modules_parallel_blocks_modules_2_modules_norm14_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_norm14_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm14_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_2_modules_norm14_parameters_bias_
        ) = None
        linear_56 = torch._C._nn.linear(
            x_237,
            l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_qkv_parameters_bias_,
        )
        x_237 = l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_qkv_parameters_bias_ = (None)
        reshape_84 = linear_56.reshape(1, 785, 3, 8, 19)
        linear_56 = None
        qkv_14 = reshape_84.permute(2, 0, 3, 1, 4)
        reshape_84 = None
        unbind_14 = qkv_14.unbind(0)
        qkv_14 = None
        q_14 = unbind_14[0]
        k_14 = unbind_14[1]
        v_14 = unbind_14[2]
        unbind_14 = None
        k_softmax_14 = k_14.softmax(dim=2)
        k_14 = None
        transpose_118 = k_softmax_14.transpose(-1, -2)
        k_softmax_14 = None
        factor_att_28 = transpose_118 @ v_14
        transpose_118 = None
        factor_att_29 = q_14 @ factor_att_28
        factor_att_28 = None
        q_img_14 = q_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_14 = None
        v_img_28 = v_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_14 = None
        transpose_119 = v_img_28.transpose(-1, -2)
        v_img_28 = None
        v_img_29 = transpose_119.reshape(1, 152, 28, 28)
        transpose_119 = None
        split_14 = torch.functional.split(v_img_29, [38, 57, 57], dim=1)
        v_img_29 = None
        getitem_179 = split_14[0]
        getitem_180 = split_14[1]
        getitem_181 = split_14[2]
        split_14 = None
        conv2d_63 = torch.conv2d(
            getitem_179,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_179 = None
        conv2d_64 = torch.conv2d(
            getitem_180,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_180 = None
        conv2d_65 = torch.conv2d(
            getitem_181,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_181 = None
        conv_v_img_28 = torch.cat([conv2d_63, conv2d_64, conv2d_65], dim=1)
        conv2d_63 = conv2d_64 = conv2d_65 = None
        reshape_86 = conv_v_img_28.reshape(1, 8, 19, 784)
        conv_v_img_28 = None
        conv_v_img_29 = reshape_86.transpose(-1, -2)
        reshape_86 = None
        EV_hat_28 = q_img_14 * conv_v_img_29
        q_img_14 = conv_v_img_29 = None
        EV_hat_29 = torch._C._nn.pad(EV_hat_28, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_28 = None
        mul_29 = 0.22941573387056177 * factor_att_29
        factor_att_29 = None
        x_240 = mul_29 + EV_hat_29
        mul_29 = EV_hat_29 = None
        transpose_121 = x_240.transpose(1, 2)
        x_240 = None
        x_241 = transpose_121.reshape(1, 785, 152)
        transpose_121 = None
        x_242 = torch._C._nn.linear(
            x_241,
            l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_proj_parameters_bias_,
        )
        x_241 = l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe2_modules_proj_parameters_bias_ = (None)
        x_243 = torch.nn.functional.dropout(x_242, 0.0, False, False)
        x_242 = None
        linear_58 = torch._C._nn.linear(
            x_238,
            l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_qkv_parameters_bias_,
        )
        x_238 = l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_qkv_parameters_bias_ = (None)
        reshape_88 = linear_58.reshape(1, 197, 3, 8, 19)
        linear_58 = None
        qkv_15 = reshape_88.permute(2, 0, 3, 1, 4)
        reshape_88 = None
        unbind_15 = qkv_15.unbind(0)
        qkv_15 = None
        q_15 = unbind_15[0]
        k_15 = unbind_15[1]
        v_15 = unbind_15[2]
        unbind_15 = None
        k_softmax_15 = k_15.softmax(dim=2)
        k_15 = None
        transpose_122 = k_softmax_15.transpose(-1, -2)
        k_softmax_15 = None
        factor_att_30 = transpose_122 @ v_15
        transpose_122 = None
        factor_att_31 = q_15 @ factor_att_30
        factor_att_30 = None
        q_img_15 = q_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_15 = None
        v_img_30 = v_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_15 = None
        transpose_123 = v_img_30.transpose(-1, -2)
        v_img_30 = None
        v_img_31 = transpose_123.reshape(1, 152, 14, 14)
        transpose_123 = None
        split_15 = torch.functional.split(v_img_31, [38, 57, 57], dim=1)
        v_img_31 = None
        getitem_187 = split_15[0]
        getitem_188 = split_15[1]
        getitem_189 = split_15[2]
        split_15 = None
        conv2d_66 = torch.conv2d(
            getitem_187,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_187 = None
        conv2d_67 = torch.conv2d(
            getitem_188,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_188 = None
        conv2d_68 = torch.conv2d(
            getitem_189,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_189 = None
        conv_v_img_30 = torch.cat([conv2d_66, conv2d_67, conv2d_68], dim=1)
        conv2d_66 = conv2d_67 = conv2d_68 = None
        reshape_90 = conv_v_img_30.reshape(1, 8, 19, 196)
        conv_v_img_30 = None
        conv_v_img_31 = reshape_90.transpose(-1, -2)
        reshape_90 = None
        EV_hat_30 = q_img_15 * conv_v_img_31
        q_img_15 = conv_v_img_31 = None
        EV_hat_31 = torch._C._nn.pad(EV_hat_30, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_30 = None
        mul_31 = 0.22941573387056177 * factor_att_31
        factor_att_31 = None
        x_244 = mul_31 + EV_hat_31
        mul_31 = EV_hat_31 = None
        transpose_125 = x_244.transpose(1, 2)
        x_244 = None
        x_245 = transpose_125.reshape(1, 197, 152)
        transpose_125 = None
        x_246 = torch._C._nn.linear(
            x_245,
            l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_proj_parameters_bias_,
        )
        x_245 = l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe3_modules_proj_parameters_bias_ = (None)
        x_247 = torch.nn.functional.dropout(x_246, 0.0, False, False)
        x_246 = None
        linear_60 = torch._C._nn.linear(
            x_239,
            l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_qkv_parameters_bias_,
        )
        x_239 = l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_qkv_parameters_bias_ = (None)
        reshape_92 = linear_60.reshape(1, 50, 3, 8, 19)
        linear_60 = None
        qkv_16 = reshape_92.permute(2, 0, 3, 1, 4)
        reshape_92 = None
        unbind_16 = qkv_16.unbind(0)
        qkv_16 = None
        q_16 = unbind_16[0]
        k_16 = unbind_16[1]
        v_16 = unbind_16[2]
        unbind_16 = None
        k_softmax_16 = k_16.softmax(dim=2)
        k_16 = None
        transpose_126 = k_softmax_16.transpose(-1, -2)
        k_softmax_16 = None
        factor_att_32 = transpose_126 @ v_16
        transpose_126 = None
        factor_att_33 = q_16 @ factor_att_32
        factor_att_32 = None
        q_img_16 = q_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_16 = None
        v_img_32 = v_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_16 = None
        transpose_127 = v_img_32.transpose(-1, -2)
        v_img_32 = None
        v_img_33 = transpose_127.reshape(1, 152, 7, 7)
        transpose_127 = None
        split_16 = torch.functional.split(v_img_33, [38, 57, 57], dim=1)
        v_img_33 = None
        getitem_195 = split_16[0]
        getitem_196 = split_16[1]
        getitem_197 = split_16[2]
        split_16 = None
        conv2d_69 = torch.conv2d(
            getitem_195,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_195 = None
        conv2d_70 = torch.conv2d(
            getitem_196,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_196 = None
        conv2d_71 = torch.conv2d(
            getitem_197,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_197 = None
        conv_v_img_32 = torch.cat([conv2d_69, conv2d_70, conv2d_71], dim=1)
        conv2d_69 = conv2d_70 = conv2d_71 = None
        reshape_94 = conv_v_img_32.reshape(1, 8, 19, 49)
        conv_v_img_32 = None
        conv_v_img_33 = reshape_94.transpose(-1, -2)
        reshape_94 = None
        EV_hat_32 = q_img_16 * conv_v_img_33
        q_img_16 = conv_v_img_33 = None
        EV_hat_33 = torch._C._nn.pad(EV_hat_32, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_32 = None
        mul_33 = 0.22941573387056177 * factor_att_33
        factor_att_33 = None
        x_248 = mul_33 + EV_hat_33
        mul_33 = EV_hat_33 = None
        transpose_129 = x_248.transpose(1, 2)
        x_248 = None
        x_249 = transpose_129.reshape(1, 50, 152)
        transpose_129 = None
        x_250 = torch._C._nn.linear(
            x_249,
            l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_proj_parameters_bias_,
        )
        x_249 = l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_2_modules_factoratt_crpe4_modules_proj_parameters_bias_ = (None)
        x_251 = torch.nn.functional.dropout(x_250, 0.0, False, False)
        x_250 = None
        cls_token_29 = x_247[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_65 = x_247[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_130 = img_tokens_65.transpose(1, 2)
        img_tokens_65 = None
        img_tokens_66 = transpose_130.reshape(1, 152, 14, 14)
        transpose_130 = None
        img_tokens_67 = torch.nn.functional.interpolate(
            img_tokens_66,
            scale_factor=2.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_66 = None
        reshape_97 = img_tokens_67.reshape(1, 152, -1)
        img_tokens_67 = None
        img_tokens_68 = reshape_97.transpose(1, 2)
        reshape_97 = None
        out_12 = torch.cat((cls_token_29, img_tokens_68), dim=1)
        cls_token_29 = img_tokens_68 = None
        cls_token_30 = x_251[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_69 = x_251[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_132 = img_tokens_69.transpose(1, 2)
        img_tokens_69 = None
        img_tokens_70 = transpose_132.reshape(1, 152, 7, 7)
        transpose_132 = None
        img_tokens_71 = torch.nn.functional.interpolate(
            img_tokens_70,
            scale_factor=2.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_70 = None
        reshape_99 = img_tokens_71.reshape(1, 152, -1)
        img_tokens_71 = None
        img_tokens_72 = reshape_99.transpose(1, 2)
        reshape_99 = None
        out_13 = torch.cat((cls_token_30, img_tokens_72), dim=1)
        cls_token_30 = img_tokens_72 = None
        cls_token_31 = x_251[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_73 = x_251[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_134 = img_tokens_73.transpose(1, 2)
        img_tokens_73 = None
        img_tokens_74 = transpose_134.reshape(1, 152, 7, 7)
        transpose_134 = None
        img_tokens_75 = torch.nn.functional.interpolate(
            img_tokens_74,
            scale_factor=4.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_74 = None
        reshape_101 = img_tokens_75.reshape(1, 152, -1)
        img_tokens_75 = None
        img_tokens_76 = reshape_101.transpose(1, 2)
        reshape_101 = None
        out_14 = torch.cat((cls_token_31, img_tokens_76), dim=1)
        cls_token_31 = img_tokens_76 = None
        cls_token_32 = x_243[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_77 = x_243[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_136 = img_tokens_77.transpose(1, 2)
        img_tokens_77 = None
        img_tokens_78 = transpose_136.reshape(1, 152, 28, 28)
        transpose_136 = None
        img_tokens_79 = torch.nn.functional.interpolate(
            img_tokens_78,
            scale_factor=0.5,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_78 = None
        reshape_103 = img_tokens_79.reshape(1, 152, -1)
        img_tokens_79 = None
        img_tokens_80 = reshape_103.transpose(1, 2)
        reshape_103 = None
        out_15 = torch.cat((cls_token_32, img_tokens_80), dim=1)
        cls_token_32 = img_tokens_80 = None
        cls_token_33 = x_247[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_81 = x_247[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_138 = img_tokens_81.transpose(1, 2)
        img_tokens_81 = None
        img_tokens_82 = transpose_138.reshape(1, 152, 14, 14)
        transpose_138 = None
        img_tokens_83 = torch.nn.functional.interpolate(
            img_tokens_82,
            scale_factor=0.5,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_82 = None
        reshape_105 = img_tokens_83.reshape(1, 152, -1)
        img_tokens_83 = None
        img_tokens_84 = reshape_105.transpose(1, 2)
        reshape_105 = None
        out_16 = torch.cat((cls_token_33, img_tokens_84), dim=1)
        cls_token_33 = img_tokens_84 = None
        cls_token_34 = x_243[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_85 = x_243[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_140 = img_tokens_85.transpose(1, 2)
        img_tokens_85 = None
        img_tokens_86 = transpose_140.reshape(1, 152, 28, 28)
        transpose_140 = None
        img_tokens_87 = torch.nn.functional.interpolate(
            img_tokens_86,
            scale_factor=0.25,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_86 = None
        reshape_107 = img_tokens_87.reshape(1, 152, -1)
        img_tokens_87 = None
        img_tokens_88 = reshape_107.transpose(1, 2)
        reshape_107 = None
        out_17 = torch.cat((cls_token_34, img_tokens_88), dim=1)
        cls_token_34 = img_tokens_88 = None
        add_74 = x_243 + out_12
        x_243 = out_12 = None
        cur2_2 = add_74 + out_14
        add_74 = out_14 = None
        add_76 = x_247 + out_13
        x_247 = out_13 = None
        cur3_2 = add_76 + out_15
        add_76 = out_15 = None
        add_78 = x_251 + out_16
        x_251 = out_16 = None
        cur4_2 = add_78 + out_17
        add_78 = out_17 = None
        x2_4 = x_230 + cur2_2
        x_230 = cur2_2 = None
        x3_4 = x_233 + cur3_2
        x_233 = cur3_2 = None
        x4_4 = x_236 + cur4_2
        x_236 = cur4_2 = None
        x_252 = torch.nn.functional.layer_norm(
            x2_4,
            (152,),
            l_self_modules_parallel_blocks_modules_2_modules_norm22_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_norm22_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm22_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_2_modules_norm22_parameters_bias_
        ) = None
        x_253 = torch.nn.functional.layer_norm(
            x3_4,
            (152,),
            l_self_modules_parallel_blocks_modules_2_modules_norm23_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_norm23_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm23_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_2_modules_norm23_parameters_bias_
        ) = None
        x_254 = torch.nn.functional.layer_norm(
            x4_4,
            (152,),
            l_self_modules_parallel_blocks_modules_2_modules_norm24_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_norm24_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_2_modules_norm24_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_2_modules_norm24_parameters_bias_
        ) = None
        x_255 = torch._C._nn.linear(
            x_252,
            l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_252 = None
        x_256 = torch._C._nn.gelu(x_255, approximate="none")
        x_255 = None
        x_257 = torch.nn.functional.dropout(x_256, 0.0, False, False)
        x_256 = None
        x_258 = torch._C._nn.linear(
            x_257,
            l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_257 = None
        x_259 = torch.nn.functional.dropout(x_258, 0.0, False, False)
        x_258 = None
        x_260 = torch._C._nn.linear(
            x_253,
            l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_253 = None
        x_261 = torch._C._nn.gelu(x_260, approximate="none")
        x_260 = None
        x_262 = torch.nn.functional.dropout(x_261, 0.0, False, False)
        x_261 = None
        x_263 = torch._C._nn.linear(
            x_262,
            l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_262 = None
        x_264 = torch.nn.functional.dropout(x_263, 0.0, False, False)
        x_263 = None
        x_265 = torch._C._nn.linear(
            x_254,
            l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_254 = l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_weight_ = l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc1_parameters_bias_ = (None)
        x_266 = torch._C._nn.gelu(x_265, approximate="none")
        x_265 = None
        x_267 = torch.nn.functional.dropout(x_266, 0.0, False, False)
        x_266 = None
        x_268 = torch._C._nn.linear(
            x_267,
            l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_267 = l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_weight_ = l_self_modules_parallel_blocks_modules_2_modules_mlp2_modules_fc2_parameters_bias_ = (None)
        x_269 = torch.nn.functional.dropout(x_268, 0.0, False, False)
        x_268 = None
        x2_5 = x2_4 + x_259
        x2_4 = x_259 = None
        x3_5 = x3_4 + x_264
        x3_4 = x_264 = None
        x4_5 = x4_4 + x_269
        x4_4 = x_269 = None
        cls_token_35 = x2_5[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_89 = x2_5[(slice(None, None, None), slice(1, None, None))]
        x2_5 = None
        transpose_142 = img_tokens_89.transpose(1, 2)
        img_tokens_89 = None
        feat_17 = transpose_142.view(1, 152, 28, 28)
        transpose_142 = None
        conv2d_72 = torch.conv2d(
            feat_17,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_270 = conv2d_72 + feat_17
        conv2d_72 = feat_17 = None
        flatten_21 = x_270.flatten(2)
        x_270 = None
        x_271 = flatten_21.transpose(1, 2)
        flatten_21 = None
        x_272 = torch.cat((cls_token_35, x_271), dim=1)
        cls_token_35 = x_271 = None
        cls_token_36 = x3_5[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_90 = x3_5[(slice(None, None, None), slice(1, None, None))]
        x3_5 = None
        transpose_144 = img_tokens_90.transpose(1, 2)
        img_tokens_90 = None
        feat_18 = transpose_144.view(1, 152, 14, 14)
        transpose_144 = None
        conv2d_73 = torch.conv2d(
            feat_18,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_273 = conv2d_73 + feat_18
        conv2d_73 = feat_18 = None
        flatten_22 = x_273.flatten(2)
        x_273 = None
        x_274 = flatten_22.transpose(1, 2)
        flatten_22 = None
        x_275 = torch.cat((cls_token_36, x_274), dim=1)
        cls_token_36 = x_274 = None
        cls_token_37 = x4_5[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_91 = x4_5[(slice(None, None, None), slice(1, None, None))]
        x4_5 = None
        transpose_146 = img_tokens_91.transpose(1, 2)
        img_tokens_91 = None
        feat_19 = transpose_146.view(1, 152, 7, 7)
        transpose_146 = None
        conv2d_74 = torch.conv2d(
            feat_19,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_276 = conv2d_74 + feat_19
        conv2d_74 = feat_19 = None
        flatten_23 = x_276.flatten(2)
        x_276 = None
        x_277 = flatten_23.transpose(1, 2)
        flatten_23 = None
        x_278 = torch.cat((cls_token_37, x_277), dim=1)
        cls_token_37 = x_277 = None
        x_279 = torch.nn.functional.layer_norm(
            x_272,
            (152,),
            l_self_modules_parallel_blocks_modules_3_modules_norm12_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_norm12_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm12_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_3_modules_norm12_parameters_bias_
        ) = None
        x_280 = torch.nn.functional.layer_norm(
            x_275,
            (152,),
            l_self_modules_parallel_blocks_modules_3_modules_norm13_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_norm13_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm13_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_3_modules_norm13_parameters_bias_
        ) = None
        x_281 = torch.nn.functional.layer_norm(
            x_278,
            (152,),
            l_self_modules_parallel_blocks_modules_3_modules_norm14_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_norm14_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm14_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_3_modules_norm14_parameters_bias_
        ) = None
        linear_68 = torch._C._nn.linear(
            x_279,
            l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_qkv_parameters_bias_,
        )
        x_279 = l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_qkv_parameters_bias_ = (None)
        reshape_108 = linear_68.reshape(1, 785, 3, 8, 19)
        linear_68 = None
        qkv_17 = reshape_108.permute(2, 0, 3, 1, 4)
        reshape_108 = None
        unbind_17 = qkv_17.unbind(0)
        qkv_17 = None
        q_17 = unbind_17[0]
        k_17 = unbind_17[1]
        v_17 = unbind_17[2]
        unbind_17 = None
        k_softmax_17 = k_17.softmax(dim=2)
        k_17 = None
        transpose_148 = k_softmax_17.transpose(-1, -2)
        k_softmax_17 = None
        factor_att_34 = transpose_148 @ v_17
        transpose_148 = None
        factor_att_35 = q_17 @ factor_att_34
        factor_att_34 = None
        q_img_17 = q_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_17 = None
        v_img_34 = v_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_17 = None
        transpose_149 = v_img_34.transpose(-1, -2)
        v_img_34 = None
        v_img_35 = transpose_149.reshape(1, 152, 28, 28)
        transpose_149 = None
        split_17 = torch.functional.split(v_img_35, [38, 57, 57], dim=1)
        v_img_35 = None
        getitem_221 = split_17[0]
        getitem_222 = split_17[1]
        getitem_223 = split_17[2]
        split_17 = None
        conv2d_75 = torch.conv2d(
            getitem_221,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_221 = None
        conv2d_76 = torch.conv2d(
            getitem_222,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_222 = None
        conv2d_77 = torch.conv2d(
            getitem_223,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_223 = None
        conv_v_img_34 = torch.cat([conv2d_75, conv2d_76, conv2d_77], dim=1)
        conv2d_75 = conv2d_76 = conv2d_77 = None
        reshape_110 = conv_v_img_34.reshape(1, 8, 19, 784)
        conv_v_img_34 = None
        conv_v_img_35 = reshape_110.transpose(-1, -2)
        reshape_110 = None
        EV_hat_34 = q_img_17 * conv_v_img_35
        q_img_17 = conv_v_img_35 = None
        EV_hat_35 = torch._C._nn.pad(EV_hat_34, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_34 = None
        mul_35 = 0.22941573387056177 * factor_att_35
        factor_att_35 = None
        x_282 = mul_35 + EV_hat_35
        mul_35 = EV_hat_35 = None
        transpose_151 = x_282.transpose(1, 2)
        x_282 = None
        x_283 = transpose_151.reshape(1, 785, 152)
        transpose_151 = None
        x_284 = torch._C._nn.linear(
            x_283,
            l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_proj_parameters_bias_,
        )
        x_283 = l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe2_modules_proj_parameters_bias_ = (None)
        x_285 = torch.nn.functional.dropout(x_284, 0.0, False, False)
        x_284 = None
        linear_70 = torch._C._nn.linear(
            x_280,
            l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_qkv_parameters_bias_,
        )
        x_280 = l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_qkv_parameters_bias_ = (None)
        reshape_112 = linear_70.reshape(1, 197, 3, 8, 19)
        linear_70 = None
        qkv_18 = reshape_112.permute(2, 0, 3, 1, 4)
        reshape_112 = None
        unbind_18 = qkv_18.unbind(0)
        qkv_18 = None
        q_18 = unbind_18[0]
        k_18 = unbind_18[1]
        v_18 = unbind_18[2]
        unbind_18 = None
        k_softmax_18 = k_18.softmax(dim=2)
        k_18 = None
        transpose_152 = k_softmax_18.transpose(-1, -2)
        k_softmax_18 = None
        factor_att_36 = transpose_152 @ v_18
        transpose_152 = None
        factor_att_37 = q_18 @ factor_att_36
        factor_att_36 = None
        q_img_18 = q_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_18 = None
        v_img_36 = v_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_18 = None
        transpose_153 = v_img_36.transpose(-1, -2)
        v_img_36 = None
        v_img_37 = transpose_153.reshape(1, 152, 14, 14)
        transpose_153 = None
        split_18 = torch.functional.split(v_img_37, [38, 57, 57], dim=1)
        v_img_37 = None
        getitem_229 = split_18[0]
        getitem_230 = split_18[1]
        getitem_231 = split_18[2]
        split_18 = None
        conv2d_78 = torch.conv2d(
            getitem_229,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_229 = None
        conv2d_79 = torch.conv2d(
            getitem_230,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_230 = None
        conv2d_80 = torch.conv2d(
            getitem_231,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_231 = None
        conv_v_img_36 = torch.cat([conv2d_78, conv2d_79, conv2d_80], dim=1)
        conv2d_78 = conv2d_79 = conv2d_80 = None
        reshape_114 = conv_v_img_36.reshape(1, 8, 19, 196)
        conv_v_img_36 = None
        conv_v_img_37 = reshape_114.transpose(-1, -2)
        reshape_114 = None
        EV_hat_36 = q_img_18 * conv_v_img_37
        q_img_18 = conv_v_img_37 = None
        EV_hat_37 = torch._C._nn.pad(EV_hat_36, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_36 = None
        mul_37 = 0.22941573387056177 * factor_att_37
        factor_att_37 = None
        x_286 = mul_37 + EV_hat_37
        mul_37 = EV_hat_37 = None
        transpose_155 = x_286.transpose(1, 2)
        x_286 = None
        x_287 = transpose_155.reshape(1, 197, 152)
        transpose_155 = None
        x_288 = torch._C._nn.linear(
            x_287,
            l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_proj_parameters_bias_,
        )
        x_287 = l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe3_modules_proj_parameters_bias_ = (None)
        x_289 = torch.nn.functional.dropout(x_288, 0.0, False, False)
        x_288 = None
        linear_72 = torch._C._nn.linear(
            x_281,
            l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_qkv_parameters_bias_,
        )
        x_281 = l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_qkv_parameters_bias_ = (None)
        reshape_116 = linear_72.reshape(1, 50, 3, 8, 19)
        linear_72 = None
        qkv_19 = reshape_116.permute(2, 0, 3, 1, 4)
        reshape_116 = None
        unbind_19 = qkv_19.unbind(0)
        qkv_19 = None
        q_19 = unbind_19[0]
        k_19 = unbind_19[1]
        v_19 = unbind_19[2]
        unbind_19 = None
        k_softmax_19 = k_19.softmax(dim=2)
        k_19 = None
        transpose_156 = k_softmax_19.transpose(-1, -2)
        k_softmax_19 = None
        factor_att_38 = transpose_156 @ v_19
        transpose_156 = None
        factor_att_39 = q_19 @ factor_att_38
        factor_att_38 = None
        q_img_19 = q_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_19 = None
        v_img_38 = v_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_19 = None
        transpose_157 = v_img_38.transpose(-1, -2)
        v_img_38 = None
        v_img_39 = transpose_157.reshape(1, 152, 7, 7)
        transpose_157 = None
        split_19 = torch.functional.split(v_img_39, [38, 57, 57], dim=1)
        v_img_39 = None
        getitem_237 = split_19[0]
        getitem_238 = split_19[1]
        getitem_239 = split_19[2]
        split_19 = None
        conv2d_81 = torch.conv2d(
            getitem_237,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_237 = None
        conv2d_82 = torch.conv2d(
            getitem_238,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_238 = None
        conv2d_83 = torch.conv2d(
            getitem_239,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_239 = None
        conv_v_img_38 = torch.cat([conv2d_81, conv2d_82, conv2d_83], dim=1)
        conv2d_81 = conv2d_82 = conv2d_83 = None
        reshape_118 = conv_v_img_38.reshape(1, 8, 19, 49)
        conv_v_img_38 = None
        conv_v_img_39 = reshape_118.transpose(-1, -2)
        reshape_118 = None
        EV_hat_38 = q_img_19 * conv_v_img_39
        q_img_19 = conv_v_img_39 = None
        EV_hat_39 = torch._C._nn.pad(EV_hat_38, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_38 = None
        mul_39 = 0.22941573387056177 * factor_att_39
        factor_att_39 = None
        x_290 = mul_39 + EV_hat_39
        mul_39 = EV_hat_39 = None
        transpose_159 = x_290.transpose(1, 2)
        x_290 = None
        x_291 = transpose_159.reshape(1, 50, 152)
        transpose_159 = None
        x_292 = torch._C._nn.linear(
            x_291,
            l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_proj_parameters_bias_,
        )
        x_291 = l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_3_modules_factoratt_crpe4_modules_proj_parameters_bias_ = (None)
        x_293 = torch.nn.functional.dropout(x_292, 0.0, False, False)
        x_292 = None
        cls_token_38 = x_289[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_92 = x_289[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_160 = img_tokens_92.transpose(1, 2)
        img_tokens_92 = None
        img_tokens_93 = transpose_160.reshape(1, 152, 14, 14)
        transpose_160 = None
        img_tokens_94 = torch.nn.functional.interpolate(
            img_tokens_93,
            scale_factor=2.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_93 = None
        reshape_121 = img_tokens_94.reshape(1, 152, -1)
        img_tokens_94 = None
        img_tokens_95 = reshape_121.transpose(1, 2)
        reshape_121 = None
        out_18 = torch.cat((cls_token_38, img_tokens_95), dim=1)
        cls_token_38 = img_tokens_95 = None
        cls_token_39 = x_293[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_96 = x_293[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_162 = img_tokens_96.transpose(1, 2)
        img_tokens_96 = None
        img_tokens_97 = transpose_162.reshape(1, 152, 7, 7)
        transpose_162 = None
        img_tokens_98 = torch.nn.functional.interpolate(
            img_tokens_97,
            scale_factor=2.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_97 = None
        reshape_123 = img_tokens_98.reshape(1, 152, -1)
        img_tokens_98 = None
        img_tokens_99 = reshape_123.transpose(1, 2)
        reshape_123 = None
        out_19 = torch.cat((cls_token_39, img_tokens_99), dim=1)
        cls_token_39 = img_tokens_99 = None
        cls_token_40 = x_293[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_100 = x_293[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_164 = img_tokens_100.transpose(1, 2)
        img_tokens_100 = None
        img_tokens_101 = transpose_164.reshape(1, 152, 7, 7)
        transpose_164 = None
        img_tokens_102 = torch.nn.functional.interpolate(
            img_tokens_101,
            scale_factor=4.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_101 = None
        reshape_125 = img_tokens_102.reshape(1, 152, -1)
        img_tokens_102 = None
        img_tokens_103 = reshape_125.transpose(1, 2)
        reshape_125 = None
        out_20 = torch.cat((cls_token_40, img_tokens_103), dim=1)
        cls_token_40 = img_tokens_103 = None
        cls_token_41 = x_285[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_104 = x_285[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_166 = img_tokens_104.transpose(1, 2)
        img_tokens_104 = None
        img_tokens_105 = transpose_166.reshape(1, 152, 28, 28)
        transpose_166 = None
        img_tokens_106 = torch.nn.functional.interpolate(
            img_tokens_105,
            scale_factor=0.5,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_105 = None
        reshape_127 = img_tokens_106.reshape(1, 152, -1)
        img_tokens_106 = None
        img_tokens_107 = reshape_127.transpose(1, 2)
        reshape_127 = None
        out_21 = torch.cat((cls_token_41, img_tokens_107), dim=1)
        cls_token_41 = img_tokens_107 = None
        cls_token_42 = x_289[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_108 = x_289[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_168 = img_tokens_108.transpose(1, 2)
        img_tokens_108 = None
        img_tokens_109 = transpose_168.reshape(1, 152, 14, 14)
        transpose_168 = None
        img_tokens_110 = torch.nn.functional.interpolate(
            img_tokens_109,
            scale_factor=0.5,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_109 = None
        reshape_129 = img_tokens_110.reshape(1, 152, -1)
        img_tokens_110 = None
        img_tokens_111 = reshape_129.transpose(1, 2)
        reshape_129 = None
        out_22 = torch.cat((cls_token_42, img_tokens_111), dim=1)
        cls_token_42 = img_tokens_111 = None
        cls_token_43 = x_285[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_112 = x_285[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_170 = img_tokens_112.transpose(1, 2)
        img_tokens_112 = None
        img_tokens_113 = transpose_170.reshape(1, 152, 28, 28)
        transpose_170 = None
        img_tokens_114 = torch.nn.functional.interpolate(
            img_tokens_113,
            scale_factor=0.25,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_113 = None
        reshape_131 = img_tokens_114.reshape(1, 152, -1)
        img_tokens_114 = None
        img_tokens_115 = reshape_131.transpose(1, 2)
        reshape_131 = None
        out_23 = torch.cat((cls_token_43, img_tokens_115), dim=1)
        cls_token_43 = img_tokens_115 = None
        add_92 = x_285 + out_18
        x_285 = out_18 = None
        cur2_3 = add_92 + out_20
        add_92 = out_20 = None
        add_94 = x_289 + out_19
        x_289 = out_19 = None
        cur3_3 = add_94 + out_21
        add_94 = out_21 = None
        add_96 = x_293 + out_22
        x_293 = out_22 = None
        cur4_3 = add_96 + out_23
        add_96 = out_23 = None
        x2_6 = x_272 + cur2_3
        x_272 = cur2_3 = None
        x3_6 = x_275 + cur3_3
        x_275 = cur3_3 = None
        x4_6 = x_278 + cur4_3
        x_278 = cur4_3 = None
        x_294 = torch.nn.functional.layer_norm(
            x2_6,
            (152,),
            l_self_modules_parallel_blocks_modules_3_modules_norm22_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_norm22_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm22_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_3_modules_norm22_parameters_bias_
        ) = None
        x_295 = torch.nn.functional.layer_norm(
            x3_6,
            (152,),
            l_self_modules_parallel_blocks_modules_3_modules_norm23_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_norm23_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm23_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_3_modules_norm23_parameters_bias_
        ) = None
        x_296 = torch.nn.functional.layer_norm(
            x4_6,
            (152,),
            l_self_modules_parallel_blocks_modules_3_modules_norm24_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_norm24_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_3_modules_norm24_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_3_modules_norm24_parameters_bias_
        ) = None
        x_297 = torch._C._nn.linear(
            x_294,
            l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_294 = None
        x_298 = torch._C._nn.gelu(x_297, approximate="none")
        x_297 = None
        x_299 = torch.nn.functional.dropout(x_298, 0.0, False, False)
        x_298 = None
        x_300 = torch._C._nn.linear(
            x_299,
            l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_299 = None
        x_301 = torch.nn.functional.dropout(x_300, 0.0, False, False)
        x_300 = None
        x_302 = torch._C._nn.linear(
            x_295,
            l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_295 = None
        x_303 = torch._C._nn.gelu(x_302, approximate="none")
        x_302 = None
        x_304 = torch.nn.functional.dropout(x_303, 0.0, False, False)
        x_303 = None
        x_305 = torch._C._nn.linear(
            x_304,
            l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_304 = None
        x_306 = torch.nn.functional.dropout(x_305, 0.0, False, False)
        x_305 = None
        x_307 = torch._C._nn.linear(
            x_296,
            l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_296 = l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_weight_ = l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc1_parameters_bias_ = (None)
        x_308 = torch._C._nn.gelu(x_307, approximate="none")
        x_307 = None
        x_309 = torch.nn.functional.dropout(x_308, 0.0, False, False)
        x_308 = None
        x_310 = torch._C._nn.linear(
            x_309,
            l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_309 = l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_weight_ = l_self_modules_parallel_blocks_modules_3_modules_mlp2_modules_fc2_parameters_bias_ = (None)
        x_311 = torch.nn.functional.dropout(x_310, 0.0, False, False)
        x_310 = None
        x2_7 = x2_6 + x_301
        x2_6 = x_301 = None
        x3_7 = x3_6 + x_306
        x3_6 = x_306 = None
        x4_7 = x4_6 + x_311
        x4_6 = x_311 = None
        cls_token_44 = x2_7[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_116 = x2_7[(slice(None, None, None), slice(1, None, None))]
        x2_7 = None
        transpose_172 = img_tokens_116.transpose(1, 2)
        img_tokens_116 = None
        feat_20 = transpose_172.view(1, 152, 28, 28)
        transpose_172 = None
        conv2d_84 = torch.conv2d(
            feat_20,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_312 = conv2d_84 + feat_20
        conv2d_84 = feat_20 = None
        flatten_24 = x_312.flatten(2)
        x_312 = None
        x_313 = flatten_24.transpose(1, 2)
        flatten_24 = None
        x_314 = torch.cat((cls_token_44, x_313), dim=1)
        cls_token_44 = x_313 = None
        cls_token_45 = x3_7[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_117 = x3_7[(slice(None, None, None), slice(1, None, None))]
        x3_7 = None
        transpose_174 = img_tokens_117.transpose(1, 2)
        img_tokens_117 = None
        feat_21 = transpose_174.view(1, 152, 14, 14)
        transpose_174 = None
        conv2d_85 = torch.conv2d(
            feat_21,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_315 = conv2d_85 + feat_21
        conv2d_85 = feat_21 = None
        flatten_25 = x_315.flatten(2)
        x_315 = None
        x_316 = flatten_25.transpose(1, 2)
        flatten_25 = None
        x_317 = torch.cat((cls_token_45, x_316), dim=1)
        cls_token_45 = x_316 = None
        cls_token_46 = x4_7[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_118 = x4_7[(slice(None, None, None), slice(1, None, None))]
        x4_7 = None
        transpose_176 = img_tokens_118.transpose(1, 2)
        img_tokens_118 = None
        feat_22 = transpose_176.view(1, 152, 7, 7)
        transpose_176 = None
        conv2d_86 = torch.conv2d(
            feat_22,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        x_318 = conv2d_86 + feat_22
        conv2d_86 = feat_22 = None
        flatten_26 = x_318.flatten(2)
        x_318 = None
        x_319 = flatten_26.transpose(1, 2)
        flatten_26 = None
        x_320 = torch.cat((cls_token_46, x_319), dim=1)
        cls_token_46 = x_319 = None
        x_321 = torch.nn.functional.layer_norm(
            x_314,
            (152,),
            l_self_modules_parallel_blocks_modules_4_modules_norm12_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_norm12_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm12_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_4_modules_norm12_parameters_bias_
        ) = None
        x_322 = torch.nn.functional.layer_norm(
            x_317,
            (152,),
            l_self_modules_parallel_blocks_modules_4_modules_norm13_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_norm13_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm13_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_4_modules_norm13_parameters_bias_
        ) = None
        x_323 = torch.nn.functional.layer_norm(
            x_320,
            (152,),
            l_self_modules_parallel_blocks_modules_4_modules_norm14_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_norm14_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm14_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_4_modules_norm14_parameters_bias_
        ) = None
        linear_80 = torch._C._nn.linear(
            x_321,
            l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_qkv_parameters_bias_,
        )
        x_321 = l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_qkv_parameters_bias_ = (None)
        reshape_132 = linear_80.reshape(1, 785, 3, 8, 19)
        linear_80 = None
        qkv_20 = reshape_132.permute(2, 0, 3, 1, 4)
        reshape_132 = None
        unbind_20 = qkv_20.unbind(0)
        qkv_20 = None
        q_20 = unbind_20[0]
        k_20 = unbind_20[1]
        v_20 = unbind_20[2]
        unbind_20 = None
        k_softmax_20 = k_20.softmax(dim=2)
        k_20 = None
        transpose_178 = k_softmax_20.transpose(-1, -2)
        k_softmax_20 = None
        factor_att_40 = transpose_178 @ v_20
        transpose_178 = None
        factor_att_41 = q_20 @ factor_att_40
        factor_att_40 = None
        q_img_20 = q_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_20 = None
        v_img_40 = v_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_20 = None
        transpose_179 = v_img_40.transpose(-1, -2)
        v_img_40 = None
        v_img_41 = transpose_179.reshape(1, 152, 28, 28)
        transpose_179 = None
        split_20 = torch.functional.split(v_img_41, [38, 57, 57], dim=1)
        v_img_41 = None
        getitem_263 = split_20[0]
        getitem_264 = split_20[1]
        getitem_265 = split_20[2]
        split_20 = None
        conv2d_87 = torch.conv2d(
            getitem_263,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_263 = None
        conv2d_88 = torch.conv2d(
            getitem_264,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_264 = None
        conv2d_89 = torch.conv2d(
            getitem_265,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_265 = None
        conv_v_img_40 = torch.cat([conv2d_87, conv2d_88, conv2d_89], dim=1)
        conv2d_87 = conv2d_88 = conv2d_89 = None
        reshape_134 = conv_v_img_40.reshape(1, 8, 19, 784)
        conv_v_img_40 = None
        conv_v_img_41 = reshape_134.transpose(-1, -2)
        reshape_134 = None
        EV_hat_40 = q_img_20 * conv_v_img_41
        q_img_20 = conv_v_img_41 = None
        EV_hat_41 = torch._C._nn.pad(EV_hat_40, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_40 = None
        mul_41 = 0.22941573387056177 * factor_att_41
        factor_att_41 = None
        x_324 = mul_41 + EV_hat_41
        mul_41 = EV_hat_41 = None
        transpose_181 = x_324.transpose(1, 2)
        x_324 = None
        x_325 = transpose_181.reshape(1, 785, 152)
        transpose_181 = None
        x_326 = torch._C._nn.linear(
            x_325,
            l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_proj_parameters_bias_,
        )
        x_325 = l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe2_modules_proj_parameters_bias_ = (None)
        x_327 = torch.nn.functional.dropout(x_326, 0.0, False, False)
        x_326 = None
        linear_82 = torch._C._nn.linear(
            x_322,
            l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_qkv_parameters_bias_,
        )
        x_322 = l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_qkv_parameters_bias_ = (None)
        reshape_136 = linear_82.reshape(1, 197, 3, 8, 19)
        linear_82 = None
        qkv_21 = reshape_136.permute(2, 0, 3, 1, 4)
        reshape_136 = None
        unbind_21 = qkv_21.unbind(0)
        qkv_21 = None
        q_21 = unbind_21[0]
        k_21 = unbind_21[1]
        v_21 = unbind_21[2]
        unbind_21 = None
        k_softmax_21 = k_21.softmax(dim=2)
        k_21 = None
        transpose_182 = k_softmax_21.transpose(-1, -2)
        k_softmax_21 = None
        factor_att_42 = transpose_182 @ v_21
        transpose_182 = None
        factor_att_43 = q_21 @ factor_att_42
        factor_att_42 = None
        q_img_21 = q_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_21 = None
        v_img_42 = v_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_21 = None
        transpose_183 = v_img_42.transpose(-1, -2)
        v_img_42 = None
        v_img_43 = transpose_183.reshape(1, 152, 14, 14)
        transpose_183 = None
        split_21 = torch.functional.split(v_img_43, [38, 57, 57], dim=1)
        v_img_43 = None
        getitem_271 = split_21[0]
        getitem_272 = split_21[1]
        getitem_273 = split_21[2]
        split_21 = None
        conv2d_90 = torch.conv2d(
            getitem_271,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_271 = None
        conv2d_91 = torch.conv2d(
            getitem_272,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_272 = None
        conv2d_92 = torch.conv2d(
            getitem_273,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_273 = None
        conv_v_img_42 = torch.cat([conv2d_90, conv2d_91, conv2d_92], dim=1)
        conv2d_90 = conv2d_91 = conv2d_92 = None
        reshape_138 = conv_v_img_42.reshape(1, 8, 19, 196)
        conv_v_img_42 = None
        conv_v_img_43 = reshape_138.transpose(-1, -2)
        reshape_138 = None
        EV_hat_42 = q_img_21 * conv_v_img_43
        q_img_21 = conv_v_img_43 = None
        EV_hat_43 = torch._C._nn.pad(EV_hat_42, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_42 = None
        mul_43 = 0.22941573387056177 * factor_att_43
        factor_att_43 = None
        x_328 = mul_43 + EV_hat_43
        mul_43 = EV_hat_43 = None
        transpose_185 = x_328.transpose(1, 2)
        x_328 = None
        x_329 = transpose_185.reshape(1, 197, 152)
        transpose_185 = None
        x_330 = torch._C._nn.linear(
            x_329,
            l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_proj_parameters_bias_,
        )
        x_329 = l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe3_modules_proj_parameters_bias_ = (None)
        x_331 = torch.nn.functional.dropout(x_330, 0.0, False, False)
        x_330 = None
        linear_84 = torch._C._nn.linear(
            x_323,
            l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_qkv_parameters_bias_,
        )
        x_323 = l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_qkv_parameters_bias_ = (None)
        reshape_140 = linear_84.reshape(1, 50, 3, 8, 19)
        linear_84 = None
        qkv_22 = reshape_140.permute(2, 0, 3, 1, 4)
        reshape_140 = None
        unbind_22 = qkv_22.unbind(0)
        qkv_22 = None
        q_22 = unbind_22[0]
        k_22 = unbind_22[1]
        v_22 = unbind_22[2]
        unbind_22 = None
        k_softmax_22 = k_22.softmax(dim=2)
        k_22 = None
        transpose_186 = k_softmax_22.transpose(-1, -2)
        k_softmax_22 = None
        factor_att_44 = transpose_186 @ v_22
        transpose_186 = None
        factor_att_45 = q_22 @ factor_att_44
        factor_att_44 = None
        q_img_22 = q_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_22 = None
        v_img_44 = v_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_22 = None
        transpose_187 = v_img_44.transpose(-1, -2)
        v_img_44 = None
        v_img_45 = transpose_187.reshape(1, 152, 7, 7)
        transpose_187 = None
        split_22 = torch.functional.split(v_img_45, [38, 57, 57], dim=1)
        v_img_45 = None
        getitem_279 = split_22[0]
        getitem_280 = split_22[1]
        getitem_281 = split_22[2]
        split_22 = None
        conv2d_93 = torch.conv2d(
            getitem_279,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_279 = None
        conv2d_94 = torch.conv2d(
            getitem_280,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_280 = None
        conv2d_95 = torch.conv2d(
            getitem_281,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_281 = None
        conv_v_img_44 = torch.cat([conv2d_93, conv2d_94, conv2d_95], dim=1)
        conv2d_93 = conv2d_94 = conv2d_95 = None
        reshape_142 = conv_v_img_44.reshape(1, 8, 19, 49)
        conv_v_img_44 = None
        conv_v_img_45 = reshape_142.transpose(-1, -2)
        reshape_142 = None
        EV_hat_44 = q_img_22 * conv_v_img_45
        q_img_22 = conv_v_img_45 = None
        EV_hat_45 = torch._C._nn.pad(EV_hat_44, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_44 = None
        mul_45 = 0.22941573387056177 * factor_att_45
        factor_att_45 = None
        x_332 = mul_45 + EV_hat_45
        mul_45 = EV_hat_45 = None
        transpose_189 = x_332.transpose(1, 2)
        x_332 = None
        x_333 = transpose_189.reshape(1, 50, 152)
        transpose_189 = None
        x_334 = torch._C._nn.linear(
            x_333,
            l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_proj_parameters_bias_,
        )
        x_333 = l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_4_modules_factoratt_crpe4_modules_proj_parameters_bias_ = (None)
        x_335 = torch.nn.functional.dropout(x_334, 0.0, False, False)
        x_334 = None
        cls_token_47 = x_331[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_119 = x_331[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_190 = img_tokens_119.transpose(1, 2)
        img_tokens_119 = None
        img_tokens_120 = transpose_190.reshape(1, 152, 14, 14)
        transpose_190 = None
        img_tokens_121 = torch.nn.functional.interpolate(
            img_tokens_120,
            scale_factor=2.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_120 = None
        reshape_145 = img_tokens_121.reshape(1, 152, -1)
        img_tokens_121 = None
        img_tokens_122 = reshape_145.transpose(1, 2)
        reshape_145 = None
        out_24 = torch.cat((cls_token_47, img_tokens_122), dim=1)
        cls_token_47 = img_tokens_122 = None
        cls_token_48 = x_335[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_123 = x_335[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_192 = img_tokens_123.transpose(1, 2)
        img_tokens_123 = None
        img_tokens_124 = transpose_192.reshape(1, 152, 7, 7)
        transpose_192 = None
        img_tokens_125 = torch.nn.functional.interpolate(
            img_tokens_124,
            scale_factor=2.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_124 = None
        reshape_147 = img_tokens_125.reshape(1, 152, -1)
        img_tokens_125 = None
        img_tokens_126 = reshape_147.transpose(1, 2)
        reshape_147 = None
        out_25 = torch.cat((cls_token_48, img_tokens_126), dim=1)
        cls_token_48 = img_tokens_126 = None
        cls_token_49 = x_335[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_127 = x_335[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_194 = img_tokens_127.transpose(1, 2)
        img_tokens_127 = None
        img_tokens_128 = transpose_194.reshape(1, 152, 7, 7)
        transpose_194 = None
        img_tokens_129 = torch.nn.functional.interpolate(
            img_tokens_128,
            scale_factor=4.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_128 = None
        reshape_149 = img_tokens_129.reshape(1, 152, -1)
        img_tokens_129 = None
        img_tokens_130 = reshape_149.transpose(1, 2)
        reshape_149 = None
        out_26 = torch.cat((cls_token_49, img_tokens_130), dim=1)
        cls_token_49 = img_tokens_130 = None
        cls_token_50 = x_327[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_131 = x_327[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_196 = img_tokens_131.transpose(1, 2)
        img_tokens_131 = None
        img_tokens_132 = transpose_196.reshape(1, 152, 28, 28)
        transpose_196 = None
        img_tokens_133 = torch.nn.functional.interpolate(
            img_tokens_132,
            scale_factor=0.5,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_132 = None
        reshape_151 = img_tokens_133.reshape(1, 152, -1)
        img_tokens_133 = None
        img_tokens_134 = reshape_151.transpose(1, 2)
        reshape_151 = None
        out_27 = torch.cat((cls_token_50, img_tokens_134), dim=1)
        cls_token_50 = img_tokens_134 = None
        cls_token_51 = x_331[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_135 = x_331[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_198 = img_tokens_135.transpose(1, 2)
        img_tokens_135 = None
        img_tokens_136 = transpose_198.reshape(1, 152, 14, 14)
        transpose_198 = None
        img_tokens_137 = torch.nn.functional.interpolate(
            img_tokens_136,
            scale_factor=0.5,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_136 = None
        reshape_153 = img_tokens_137.reshape(1, 152, -1)
        img_tokens_137 = None
        img_tokens_138 = reshape_153.transpose(1, 2)
        reshape_153 = None
        out_28 = torch.cat((cls_token_51, img_tokens_138), dim=1)
        cls_token_51 = img_tokens_138 = None
        cls_token_52 = x_327[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_139 = x_327[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_200 = img_tokens_139.transpose(1, 2)
        img_tokens_139 = None
        img_tokens_140 = transpose_200.reshape(1, 152, 28, 28)
        transpose_200 = None
        img_tokens_141 = torch.nn.functional.interpolate(
            img_tokens_140,
            scale_factor=0.25,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_140 = None
        reshape_155 = img_tokens_141.reshape(1, 152, -1)
        img_tokens_141 = None
        img_tokens_142 = reshape_155.transpose(1, 2)
        reshape_155 = None
        out_29 = torch.cat((cls_token_52, img_tokens_142), dim=1)
        cls_token_52 = img_tokens_142 = None
        add_110 = x_327 + out_24
        x_327 = out_24 = None
        cur2_4 = add_110 + out_26
        add_110 = out_26 = None
        add_112 = x_331 + out_25
        x_331 = out_25 = None
        cur3_4 = add_112 + out_27
        add_112 = out_27 = None
        add_114 = x_335 + out_28
        x_335 = out_28 = None
        cur4_4 = add_114 + out_29
        add_114 = out_29 = None
        x2_8 = x_314 + cur2_4
        x_314 = cur2_4 = None
        x3_8 = x_317 + cur3_4
        x_317 = cur3_4 = None
        x4_8 = x_320 + cur4_4
        x_320 = cur4_4 = None
        x_336 = torch.nn.functional.layer_norm(
            x2_8,
            (152,),
            l_self_modules_parallel_blocks_modules_4_modules_norm22_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_norm22_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm22_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_4_modules_norm22_parameters_bias_
        ) = None
        x_337 = torch.nn.functional.layer_norm(
            x3_8,
            (152,),
            l_self_modules_parallel_blocks_modules_4_modules_norm23_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_norm23_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm23_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_4_modules_norm23_parameters_bias_
        ) = None
        x_338 = torch.nn.functional.layer_norm(
            x4_8,
            (152,),
            l_self_modules_parallel_blocks_modules_4_modules_norm24_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_norm24_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_4_modules_norm24_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_4_modules_norm24_parameters_bias_
        ) = None
        x_339 = torch._C._nn.linear(
            x_336,
            l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_336 = None
        x_340 = torch._C._nn.gelu(x_339, approximate="none")
        x_339 = None
        x_341 = torch.nn.functional.dropout(x_340, 0.0, False, False)
        x_340 = None
        x_342 = torch._C._nn.linear(
            x_341,
            l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_341 = None
        x_343 = torch.nn.functional.dropout(x_342, 0.0, False, False)
        x_342 = None
        x_344 = torch._C._nn.linear(
            x_337,
            l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_337 = None
        x_345 = torch._C._nn.gelu(x_344, approximate="none")
        x_344 = None
        x_346 = torch.nn.functional.dropout(x_345, 0.0, False, False)
        x_345 = None
        x_347 = torch._C._nn.linear(
            x_346,
            l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_346 = None
        x_348 = torch.nn.functional.dropout(x_347, 0.0, False, False)
        x_347 = None
        x_349 = torch._C._nn.linear(
            x_338,
            l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_338 = l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_weight_ = l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc1_parameters_bias_ = (None)
        x_350 = torch._C._nn.gelu(x_349, approximate="none")
        x_349 = None
        x_351 = torch.nn.functional.dropout(x_350, 0.0, False, False)
        x_350 = None
        x_352 = torch._C._nn.linear(
            x_351,
            l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_351 = l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_weight_ = l_self_modules_parallel_blocks_modules_4_modules_mlp2_modules_fc2_parameters_bias_ = (None)
        x_353 = torch.nn.functional.dropout(x_352, 0.0, False, False)
        x_352 = None
        x2_9 = x2_8 + x_343
        x2_8 = x_343 = None
        x3_9 = x3_8 + x_348
        x3_8 = x_348 = None
        x4_9 = x4_8 + x_353
        x4_8 = x_353 = None
        cls_token_53 = x2_9[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_143 = x2_9[(slice(None, None, None), slice(1, None, None))]
        x2_9 = None
        transpose_202 = img_tokens_143.transpose(1, 2)
        img_tokens_143 = None
        feat_23 = transpose_202.view(1, 152, 28, 28)
        transpose_202 = None
        conv2d_96 = torch.conv2d(
            feat_23,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_ = (None)
        x_354 = conv2d_96 + feat_23
        conv2d_96 = feat_23 = None
        flatten_27 = x_354.flatten(2)
        x_354 = None
        x_355 = flatten_27.transpose(1, 2)
        flatten_27 = None
        x_356 = torch.cat((cls_token_53, x_355), dim=1)
        cls_token_53 = x_355 = None
        cls_token_54 = x3_9[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_144 = x3_9[(slice(None, None, None), slice(1, None, None))]
        x3_9 = None
        transpose_204 = img_tokens_144.transpose(1, 2)
        img_tokens_144 = None
        feat_24 = transpose_204.view(1, 152, 14, 14)
        transpose_204 = None
        conv2d_97 = torch.conv2d(
            feat_24,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_ = (None)
        x_357 = conv2d_97 + feat_24
        conv2d_97 = feat_24 = None
        flatten_28 = x_357.flatten(2)
        x_357 = None
        x_358 = flatten_28.transpose(1, 2)
        flatten_28 = None
        x_359 = torch.cat((cls_token_54, x_358), dim=1)
        cls_token_54 = x_358 = None
        cls_token_55 = x4_9[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_145 = x4_9[(slice(None, None, None), slice(1, None, None))]
        x4_9 = None
        transpose_206 = img_tokens_145.transpose(1, 2)
        img_tokens_145 = None
        feat_25 = transpose_206.view(1, 152, 7, 7)
        transpose_206 = None
        conv2d_98 = torch.conv2d(
            feat_25,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            152,
        )
        l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_ = (None)
        x_360 = conv2d_98 + feat_25
        conv2d_98 = feat_25 = None
        flatten_29 = x_360.flatten(2)
        x_360 = None
        x_361 = flatten_29.transpose(1, 2)
        flatten_29 = None
        x_362 = torch.cat((cls_token_55, x_361), dim=1)
        cls_token_55 = x_361 = None
        x_363 = torch.nn.functional.layer_norm(
            x_356,
            (152,),
            l_self_modules_parallel_blocks_modules_5_modules_norm12_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_norm12_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm12_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_5_modules_norm12_parameters_bias_
        ) = None
        x_364 = torch.nn.functional.layer_norm(
            x_359,
            (152,),
            l_self_modules_parallel_blocks_modules_5_modules_norm13_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_norm13_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm13_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_5_modules_norm13_parameters_bias_
        ) = None
        x_365 = torch.nn.functional.layer_norm(
            x_362,
            (152,),
            l_self_modules_parallel_blocks_modules_5_modules_norm14_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_norm14_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm14_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_5_modules_norm14_parameters_bias_
        ) = None
        linear_92 = torch._C._nn.linear(
            x_363,
            l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_qkv_parameters_bias_,
        )
        x_363 = l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_qkv_parameters_bias_ = (None)
        reshape_156 = linear_92.reshape(1, 785, 3, 8, 19)
        linear_92 = None
        qkv_23 = reshape_156.permute(2, 0, 3, 1, 4)
        reshape_156 = None
        unbind_23 = qkv_23.unbind(0)
        qkv_23 = None
        q_23 = unbind_23[0]
        k_23 = unbind_23[1]
        v_23 = unbind_23[2]
        unbind_23 = None
        k_softmax_23 = k_23.softmax(dim=2)
        k_23 = None
        transpose_208 = k_softmax_23.transpose(-1, -2)
        k_softmax_23 = None
        factor_att_46 = transpose_208 @ v_23
        transpose_208 = None
        factor_att_47 = q_23 @ factor_att_46
        factor_att_46 = None
        q_img_23 = q_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_23 = None
        v_img_46 = v_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_23 = None
        transpose_209 = v_img_46.transpose(-1, -2)
        v_img_46 = None
        v_img_47 = transpose_209.reshape(1, 152, 28, 28)
        transpose_209 = None
        split_23 = torch.functional.split(v_img_47, [38, 57, 57], dim=1)
        v_img_47 = None
        getitem_305 = split_23[0]
        getitem_306 = split_23[1]
        getitem_307 = split_23[2]
        split_23 = None
        conv2d_99 = torch.conv2d(
            getitem_305,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_305 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_100 = torch.conv2d(
            getitem_306,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_306 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_101 = torch.conv2d(
            getitem_307,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_307 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
        conv_v_img_46 = torch.cat([conv2d_99, conv2d_100, conv2d_101], dim=1)
        conv2d_99 = conv2d_100 = conv2d_101 = None
        reshape_158 = conv_v_img_46.reshape(1, 8, 19, 784)
        conv_v_img_46 = None
        conv_v_img_47 = reshape_158.transpose(-1, -2)
        reshape_158 = None
        EV_hat_46 = q_img_23 * conv_v_img_47
        q_img_23 = conv_v_img_47 = None
        EV_hat_47 = torch._C._nn.pad(EV_hat_46, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_46 = None
        mul_47 = 0.22941573387056177 * factor_att_47
        factor_att_47 = None
        x_366 = mul_47 + EV_hat_47
        mul_47 = EV_hat_47 = None
        transpose_211 = x_366.transpose(1, 2)
        x_366 = None
        x_367 = transpose_211.reshape(1, 785, 152)
        transpose_211 = None
        x_368 = torch._C._nn.linear(
            x_367,
            l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_proj_parameters_bias_,
        )
        x_367 = l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe2_modules_proj_parameters_bias_ = (None)
        x_369 = torch.nn.functional.dropout(x_368, 0.0, False, False)
        x_368 = None
        linear_94 = torch._C._nn.linear(
            x_364,
            l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_qkv_parameters_bias_,
        )
        x_364 = l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_qkv_parameters_bias_ = (None)
        reshape_160 = linear_94.reshape(1, 197, 3, 8, 19)
        linear_94 = None
        qkv_24 = reshape_160.permute(2, 0, 3, 1, 4)
        reshape_160 = None
        unbind_24 = qkv_24.unbind(0)
        qkv_24 = None
        q_24 = unbind_24[0]
        k_24 = unbind_24[1]
        v_24 = unbind_24[2]
        unbind_24 = None
        k_softmax_24 = k_24.softmax(dim=2)
        k_24 = None
        transpose_212 = k_softmax_24.transpose(-1, -2)
        k_softmax_24 = None
        factor_att_48 = transpose_212 @ v_24
        transpose_212 = None
        factor_att_49 = q_24 @ factor_att_48
        factor_att_48 = None
        q_img_24 = q_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_24 = None
        v_img_48 = v_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_24 = None
        transpose_213 = v_img_48.transpose(-1, -2)
        v_img_48 = None
        v_img_49 = transpose_213.reshape(1, 152, 14, 14)
        transpose_213 = None
        split_24 = torch.functional.split(v_img_49, [38, 57, 57], dim=1)
        v_img_49 = None
        getitem_313 = split_24[0]
        getitem_314 = split_24[1]
        getitem_315 = split_24[2]
        split_24 = None
        conv2d_102 = torch.conv2d(
            getitem_313,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_313 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_103 = torch.conv2d(
            getitem_314,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_314 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_104 = torch.conv2d(
            getitem_315,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_315 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
        conv_v_img_48 = torch.cat([conv2d_102, conv2d_103, conv2d_104], dim=1)
        conv2d_102 = conv2d_103 = conv2d_104 = None
        reshape_162 = conv_v_img_48.reshape(1, 8, 19, 196)
        conv_v_img_48 = None
        conv_v_img_49 = reshape_162.transpose(-1, -2)
        reshape_162 = None
        EV_hat_48 = q_img_24 * conv_v_img_49
        q_img_24 = conv_v_img_49 = None
        EV_hat_49 = torch._C._nn.pad(EV_hat_48, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_48 = None
        mul_49 = 0.22941573387056177 * factor_att_49
        factor_att_49 = None
        x_370 = mul_49 + EV_hat_49
        mul_49 = EV_hat_49 = None
        transpose_215 = x_370.transpose(1, 2)
        x_370 = None
        x_371 = transpose_215.reshape(1, 197, 152)
        transpose_215 = None
        x_372 = torch._C._nn.linear(
            x_371,
            l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_proj_parameters_bias_,
        )
        x_371 = l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe3_modules_proj_parameters_bias_ = (None)
        x_373 = torch.nn.functional.dropout(x_372, 0.0, False, False)
        x_372 = None
        linear_96 = torch._C._nn.linear(
            x_365,
            l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_qkv_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_qkv_parameters_bias_,
        )
        x_365 = l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_qkv_parameters_weight_ = l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_qkv_parameters_bias_ = (None)
        reshape_164 = linear_96.reshape(1, 50, 3, 8, 19)
        linear_96 = None
        qkv_25 = reshape_164.permute(2, 0, 3, 1, 4)
        reshape_164 = None
        unbind_25 = qkv_25.unbind(0)
        qkv_25 = None
        q_25 = unbind_25[0]
        k_25 = unbind_25[1]
        v_25 = unbind_25[2]
        unbind_25 = None
        k_softmax_25 = k_25.softmax(dim=2)
        k_25 = None
        transpose_216 = k_softmax_25.transpose(-1, -2)
        k_softmax_25 = None
        factor_att_50 = transpose_216 @ v_25
        transpose_216 = None
        factor_att_51 = q_25 @ factor_att_50
        factor_att_50 = None
        q_img_25 = q_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_25 = None
        v_img_50 = v_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_25 = None
        transpose_217 = v_img_50.transpose(-1, -2)
        v_img_50 = None
        v_img_51 = transpose_217.reshape(1, 152, 7, 7)
        transpose_217 = None
        split_25 = torch.functional.split(v_img_51, [38, 57, 57], dim=1)
        v_img_51 = None
        getitem_321 = split_25[0]
        getitem_322 = split_25[1]
        getitem_323 = split_25[2]
        split_25 = None
        conv2d_105 = torch.conv2d(
            getitem_321,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        getitem_321 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_106 = torch.conv2d(
            getitem_322,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            57,
        )
        getitem_322 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_107 = torch.conv2d(
            getitem_323,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            57,
        )
        getitem_323 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
        conv_v_img_50 = torch.cat([conv2d_105, conv2d_106, conv2d_107], dim=1)
        conv2d_105 = conv2d_106 = conv2d_107 = None
        reshape_166 = conv_v_img_50.reshape(1, 8, 19, 49)
        conv_v_img_50 = None
        conv_v_img_51 = reshape_166.transpose(-1, -2)
        reshape_166 = None
        EV_hat_50 = q_img_25 * conv_v_img_51
        q_img_25 = conv_v_img_51 = None
        EV_hat_51 = torch._C._nn.pad(EV_hat_50, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_50 = None
        mul_51 = 0.22941573387056177 * factor_att_51
        factor_att_51 = None
        x_374 = mul_51 + EV_hat_51
        mul_51 = EV_hat_51 = None
        transpose_219 = x_374.transpose(1, 2)
        x_374 = None
        x_375 = transpose_219.reshape(1, 50, 152)
        transpose_219 = None
        x_376 = torch._C._nn.linear(
            x_375,
            l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_proj_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_proj_parameters_bias_,
        )
        x_375 = l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_proj_parameters_weight_ = l_self_modules_parallel_blocks_modules_5_modules_factoratt_crpe4_modules_proj_parameters_bias_ = (None)
        x_377 = torch.nn.functional.dropout(x_376, 0.0, False, False)
        x_376 = None
        cls_token_56 = x_373[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_146 = x_373[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_220 = img_tokens_146.transpose(1, 2)
        img_tokens_146 = None
        img_tokens_147 = transpose_220.reshape(1, 152, 14, 14)
        transpose_220 = None
        img_tokens_148 = torch.nn.functional.interpolate(
            img_tokens_147,
            scale_factor=2.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_147 = None
        reshape_169 = img_tokens_148.reshape(1, 152, -1)
        img_tokens_148 = None
        img_tokens_149 = reshape_169.transpose(1, 2)
        reshape_169 = None
        out_30 = torch.cat((cls_token_56, img_tokens_149), dim=1)
        cls_token_56 = img_tokens_149 = None
        cls_token_57 = x_377[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_150 = x_377[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_222 = img_tokens_150.transpose(1, 2)
        img_tokens_150 = None
        img_tokens_151 = transpose_222.reshape(1, 152, 7, 7)
        transpose_222 = None
        img_tokens_152 = torch.nn.functional.interpolate(
            img_tokens_151,
            scale_factor=2.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_151 = None
        reshape_171 = img_tokens_152.reshape(1, 152, -1)
        img_tokens_152 = None
        img_tokens_153 = reshape_171.transpose(1, 2)
        reshape_171 = None
        out_31 = torch.cat((cls_token_57, img_tokens_153), dim=1)
        cls_token_57 = img_tokens_153 = None
        cls_token_58 = x_377[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_154 = x_377[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_224 = img_tokens_154.transpose(1, 2)
        img_tokens_154 = None
        img_tokens_155 = transpose_224.reshape(1, 152, 7, 7)
        transpose_224 = None
        img_tokens_156 = torch.nn.functional.interpolate(
            img_tokens_155,
            scale_factor=4.0,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_155 = None
        reshape_173 = img_tokens_156.reshape(1, 152, -1)
        img_tokens_156 = None
        img_tokens_157 = reshape_173.transpose(1, 2)
        reshape_173 = None
        out_32 = torch.cat((cls_token_58, img_tokens_157), dim=1)
        cls_token_58 = img_tokens_157 = None
        cls_token_59 = x_369[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_158 = x_369[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_226 = img_tokens_158.transpose(1, 2)
        img_tokens_158 = None
        img_tokens_159 = transpose_226.reshape(1, 152, 28, 28)
        transpose_226 = None
        img_tokens_160 = torch.nn.functional.interpolate(
            img_tokens_159,
            scale_factor=0.5,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_159 = None
        reshape_175 = img_tokens_160.reshape(1, 152, -1)
        img_tokens_160 = None
        img_tokens_161 = reshape_175.transpose(1, 2)
        reshape_175 = None
        out_33 = torch.cat((cls_token_59, img_tokens_161), dim=1)
        cls_token_59 = img_tokens_161 = None
        cls_token_60 = x_373[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_162 = x_373[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_228 = img_tokens_162.transpose(1, 2)
        img_tokens_162 = None
        img_tokens_163 = transpose_228.reshape(1, 152, 14, 14)
        transpose_228 = None
        img_tokens_164 = torch.nn.functional.interpolate(
            img_tokens_163,
            scale_factor=0.5,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_163 = None
        reshape_177 = img_tokens_164.reshape(1, 152, -1)
        img_tokens_164 = None
        img_tokens_165 = reshape_177.transpose(1, 2)
        reshape_177 = None
        out_34 = torch.cat((cls_token_60, img_tokens_165), dim=1)
        cls_token_60 = img_tokens_165 = None
        cls_token_61 = x_369[
            (slice(None, None, None), slice(None, 1, None), slice(None, None, None))
        ]
        img_tokens_166 = x_369[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        transpose_230 = img_tokens_166.transpose(1, 2)
        img_tokens_166 = None
        img_tokens_167 = transpose_230.reshape(1, 152, 28, 28)
        transpose_230 = None
        img_tokens_168 = torch.nn.functional.interpolate(
            img_tokens_167,
            scale_factor=0.25,
            recompute_scale_factor=False,
            mode="bilinear",
            align_corners=False,
        )
        img_tokens_167 = None
        reshape_179 = img_tokens_168.reshape(1, 152, -1)
        img_tokens_168 = None
        img_tokens_169 = reshape_179.transpose(1, 2)
        reshape_179 = None
        out_35 = torch.cat((cls_token_61, img_tokens_169), dim=1)
        cls_token_61 = img_tokens_169 = None
        add_128 = x_369 + out_30
        x_369 = out_30 = None
        cur2_5 = add_128 + out_32
        add_128 = out_32 = None
        add_130 = x_373 + out_31
        x_373 = out_31 = None
        cur3_5 = add_130 + out_33
        add_130 = out_33 = None
        add_132 = x_377 + out_34
        x_377 = out_34 = None
        cur4_5 = add_132 + out_35
        add_132 = out_35 = None
        x2_10 = x_356 + cur2_5
        x_356 = cur2_5 = None
        x3_10 = x_359 + cur3_5
        x_359 = cur3_5 = None
        x4_10 = x_362 + cur4_5
        x_362 = cur4_5 = None
        x_378 = torch.nn.functional.layer_norm(
            x2_10,
            (152,),
            l_self_modules_parallel_blocks_modules_5_modules_norm22_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_norm22_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm22_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_5_modules_norm22_parameters_bias_
        ) = None
        x_379 = torch.nn.functional.layer_norm(
            x3_10,
            (152,),
            l_self_modules_parallel_blocks_modules_5_modules_norm23_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_norm23_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm23_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_5_modules_norm23_parameters_bias_
        ) = None
        x_380 = torch.nn.functional.layer_norm(
            x4_10,
            (152,),
            l_self_modules_parallel_blocks_modules_5_modules_norm24_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_norm24_parameters_bias_,
            1e-06,
        )
        l_self_modules_parallel_blocks_modules_5_modules_norm24_parameters_weight_ = (
            l_self_modules_parallel_blocks_modules_5_modules_norm24_parameters_bias_
        ) = None
        x_381 = torch._C._nn.linear(
            x_378,
            l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_378 = None
        x_382 = torch._C._nn.gelu(x_381, approximate="none")
        x_381 = None
        x_383 = torch.nn.functional.dropout(x_382, 0.0, False, False)
        x_382 = None
        x_384 = torch._C._nn.linear(
            x_383,
            l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_383 = None
        x_385 = torch.nn.functional.dropout(x_384, 0.0, False, False)
        x_384 = None
        x_386 = torch._C._nn.linear(
            x_379,
            l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_379 = None
        x_387 = torch._C._nn.gelu(x_386, approximate="none")
        x_386 = None
        x_388 = torch.nn.functional.dropout(x_387, 0.0, False, False)
        x_387 = None
        x_389 = torch._C._nn.linear(
            x_388,
            l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_388 = None
        x_390 = torch.nn.functional.dropout(x_389, 0.0, False, False)
        x_389 = None
        x_391 = torch._C._nn.linear(
            x_380,
            l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_bias_,
        )
        x_380 = l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_weight_ = l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc1_parameters_bias_ = (None)
        x_392 = torch._C._nn.gelu(x_391, approximate="none")
        x_391 = None
        x_393 = torch.nn.functional.dropout(x_392, 0.0, False, False)
        x_392 = None
        x_394 = torch._C._nn.linear(
            x_393,
            l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_weight_,
            l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_bias_,
        )
        x_393 = l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_weight_ = l_self_modules_parallel_blocks_modules_5_modules_mlp2_modules_fc2_parameters_bias_ = (None)
        x_395 = torch.nn.functional.dropout(x_394, 0.0, False, False)
        x_394 = None
        x2_11 = x2_10 + x_385
        x2_10 = x_385 = None
        x3_11 = x3_10 + x_390
        x3_10 = x_390 = None
        x4_11 = x4_10 + x_395
        x4_10 = x_395 = None
        x_396 = torch.nn.functional.layer_norm(
            x2_11,
            (152,),
            l_self_modules_norm2_parameters_weight_,
            l_self_modules_norm2_parameters_bias_,
            1e-06,
        )
        x2_11 = (
            l_self_modules_norm2_parameters_weight_
        ) = l_self_modules_norm2_parameters_bias_ = None
        x_397 = torch.nn.functional.layer_norm(
            x3_11,
            (152,),
            l_self_modules_norm3_parameters_weight_,
            l_self_modules_norm3_parameters_bias_,
            1e-06,
        )
        x3_11 = (
            l_self_modules_norm3_parameters_weight_
        ) = l_self_modules_norm3_parameters_bias_ = None
        x_398 = torch.nn.functional.layer_norm(
            x4_11,
            (152,),
            l_self_modules_norm4_parameters_weight_,
            l_self_modules_norm4_parameters_bias_,
            1e-06,
        )
        x4_11 = (
            l_self_modules_norm4_parameters_weight_
        ) = l_self_modules_norm4_parameters_bias_ = None
        getitem_336 = x_396[(slice(None, None, None), 0)]
        x_396 = None
        getitem_337 = x_397[(slice(None, None, None), 0)]
        x_397 = None
        getitem_338 = x_398[(slice(None, None, None), 0)]
        x_398 = None
        x_399 = torch.stack([getitem_336, getitem_337, getitem_338], dim=1)
        getitem_336 = getitem_337 = getitem_338 = None
        conv1d = torch.conv1d(
            x_399,
            l_self_modules_aggregate_parameters_weight_,
            l_self_modules_aggregate_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_399 = (
            l_self_modules_aggregate_parameters_weight_
        ) = l_self_modules_aggregate_parameters_bias_ = None
        x_400 = conv1d.squeeze(dim=1)
        conv1d = None
        x_401 = torch.nn.functional.dropout(x_400, 0.0, False, False)
        x_400 = None
        x_402 = torch._C._nn.linear(
            x_401,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_401 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_402,)
