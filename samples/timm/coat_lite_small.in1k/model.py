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
        L_self_modules_serial_blocks1_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_serial_blocks2_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm4_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_serial_blocks1_modules_2_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks1_modules_2_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks1_modules_2_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks1_modules_2_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks1_modules_2_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks1_modules_2_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks1_modules_2_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks1_modules_2_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_serial_blocks2_modules_2_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks2_modules_2_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks2_modules_2_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks2_modules_2_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks2_modules_2_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks2_modules_2_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks2_modules_2_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks2_modules_2_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks2_modules_3_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks2_modules_3_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks2_modules_3_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks2_modules_3_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks2_modules_3_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks2_modules_3_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks2_modules_3_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks2_modules_3_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_norm4_parameters_weight_ = (
            L_self_modules_norm4_parameters_weight_
        )
        l_self_modules_norm4_parameters_bias_ = L_self_modules_norm4_parameters_bias_
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
            (64,),
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
        feat = transpose_1.view(1, 64, 56, 56)
        transpose_1 = None
        conv2d_1 = torch.conv2d(
            feat,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
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
            (64,),
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
        reshape = linear.reshape(1, 3137, 3, 8, 8)
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
        v_img_1 = transpose_4.reshape(1, 64, 56, 56)
        transpose_4 = None
        split = torch.functional.split(v_img_1, [16, 24, 24], dim=1)
        v_img_1 = None
        getitem_15 = split[0]
        getitem_16 = split[1]
        getitem_17 = split[2]
        split = None
        conv2d_2 = torch.conv2d(
            getitem_15,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            16,
        )
        getitem_15 = None
        conv2d_3 = torch.conv2d(
            getitem_16,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            24,
        )
        getitem_16 = None
        conv2d_4 = torch.conv2d(
            getitem_17,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            24,
        )
        getitem_17 = None
        conv_v_img = torch.cat([conv2d_2, conv2d_3, conv2d_4], dim=1)
        conv2d_2 = conv2d_3 = conv2d_4 = None
        reshape_2 = conv_v_img.reshape(1, 8, 8, 3136)
        conv_v_img = None
        conv_v_img_1 = reshape_2.transpose(-1, -2)
        reshape_2 = None
        EV_hat = q_img * conv_v_img_1
        q_img = conv_v_img_1 = None
        EV_hat_1 = torch._C._nn.pad(EV_hat, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat = None
        mul_1 = 0.3535533905932738 * factor_att_1
        factor_att_1 = None
        x_8 = mul_1 + EV_hat_1
        mul_1 = EV_hat_1 = None
        transpose_6 = x_8.transpose(1, 2)
        x_8 = None
        x_9 = transpose_6.reshape(1, 3137, 64)
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
            (64,),
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
        feat_1 = transpose_7.view(1, 64, 56, 56)
        transpose_7 = None
        conv2d_5 = torch.conv2d(
            feat_1,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
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
            (64,),
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
        reshape_4 = linear_4.reshape(1, 3137, 3, 8, 8)
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
        v_img_3 = transpose_10.reshape(1, 64, 56, 56)
        transpose_10 = None
        split_1 = torch.functional.split(v_img_3, [16, 24, 24], dim=1)
        v_img_3 = None
        getitem_25 = split_1[0]
        getitem_26 = split_1[1]
        getitem_27 = split_1[2]
        split_1 = None
        conv2d_6 = torch.conv2d(
            getitem_25,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            16,
        )
        getitem_25 = None
        conv2d_7 = torch.conv2d(
            getitem_26,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            24,
        )
        getitem_26 = None
        conv2d_8 = torch.conv2d(
            getitem_27,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            24,
        )
        getitem_27 = None
        conv_v_img_2 = torch.cat([conv2d_6, conv2d_7, conv2d_8], dim=1)
        conv2d_6 = conv2d_7 = conv2d_8 = None
        reshape_6 = conv_v_img_2.reshape(1, 8, 8, 3136)
        conv_v_img_2 = None
        conv_v_img_3 = reshape_6.transpose(-1, -2)
        reshape_6 = None
        EV_hat_2 = q_img_1 * conv_v_img_3
        q_img_1 = conv_v_img_3 = None
        EV_hat_3 = torch._C._nn.pad(EV_hat_2, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_2 = None
        mul_3 = 0.3535533905932738 * factor_att_3
        factor_att_3 = None
        x_24 = mul_3 + EV_hat_3
        mul_3 = EV_hat_3 = None
        transpose_12 = x_24.transpose(1, 2)
        x_24 = None
        x_25 = transpose_12.reshape(1, 3137, 64)
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
            (64,),
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
        cls_token_2 = x_35[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_2 = x_35[(slice(None, None, None), slice(1, None, None))]
        x_35 = None
        transpose_13 = img_tokens_2.transpose(1, 2)
        img_tokens_2 = None
        feat_2 = transpose_13.view(1, 64, 56, 56)
        transpose_13 = None
        conv2d_9 = torch.conv2d(
            feat_2,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_ = (None)
        x_36 = conv2d_9 + feat_2
        conv2d_9 = feat_2 = None
        flatten_3 = x_36.flatten(2)
        x_36 = None
        x_37 = flatten_3.transpose(1, 2)
        flatten_3 = None
        x_38 = torch.cat((cls_token_2, x_37), dim=1)
        cls_token_2 = x_37 = None
        x_39 = torch.nn.functional.layer_norm(
            x_38,
            (64,),
            l_self_modules_serial_blocks1_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks1_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks1_modules_2_modules_norm1_parameters_bias_
        ) = None
        linear_8 = torch._C._nn.linear(
            x_39,
            l_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_39 = l_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_8 = linear_8.reshape(1, 3137, 3, 8, 8)
        linear_8 = None
        qkv_2 = reshape_8.permute(2, 0, 3, 1, 4)
        reshape_8 = None
        unbind_2 = qkv_2.unbind(0)
        qkv_2 = None
        q_2 = unbind_2[0]
        k_2 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        k_softmax_2 = k_2.softmax(dim=2)
        k_2 = None
        transpose_15 = k_softmax_2.transpose(-1, -2)
        k_softmax_2 = None
        factor_att_4 = transpose_15 @ v_2
        transpose_15 = None
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
        transpose_16 = v_img_4.transpose(-1, -2)
        v_img_4 = None
        v_img_5 = transpose_16.reshape(1, 64, 56, 56)
        transpose_16 = None
        split_2 = torch.functional.split(v_img_5, [16, 24, 24], dim=1)
        v_img_5 = None
        getitem_35 = split_2[0]
        getitem_36 = split_2[1]
        getitem_37 = split_2[2]
        split_2 = None
        conv2d_10 = torch.conv2d(
            getitem_35,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            16,
        )
        getitem_35 = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_11 = torch.conv2d(
            getitem_36,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            24,
        )
        getitem_36 = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_12 = torch.conv2d(
            getitem_37,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            24,
        )
        getitem_37 = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
        conv_v_img_4 = torch.cat([conv2d_10, conv2d_11, conv2d_12], dim=1)
        conv2d_10 = conv2d_11 = conv2d_12 = None
        reshape_10 = conv_v_img_4.reshape(1, 8, 8, 3136)
        conv_v_img_4 = None
        conv_v_img_5 = reshape_10.transpose(-1, -2)
        reshape_10 = None
        EV_hat_4 = q_img_2 * conv_v_img_5
        q_img_2 = conv_v_img_5 = None
        EV_hat_5 = torch._C._nn.pad(EV_hat_4, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_4 = None
        mul_5 = 0.3535533905932738 * factor_att_5
        factor_att_5 = None
        x_40 = mul_5 + EV_hat_5
        mul_5 = EV_hat_5 = None
        transpose_18 = x_40.transpose(1, 2)
        x_40 = None
        x_41 = transpose_18.reshape(1, 3137, 64)
        transpose_18 = None
        x_42 = torch._C._nn.linear(
            x_41,
            l_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_41 = l_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks1_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_43 = torch.nn.functional.dropout(x_42, 0.0, False, False)
        x_42 = None
        x_44 = x_38 + x_43
        x_38 = x_43 = None
        x_45 = torch.nn.functional.layer_norm(
            x_44,
            (64,),
            l_self_modules_serial_blocks1_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks1_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks1_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_46 = torch._C._nn.linear(
            x_45,
            l_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_45 = l_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_47 = torch._C._nn.gelu(x_46, approximate="none")
        x_46 = None
        x_48 = torch.nn.functional.dropout(x_47, 0.0, False, False)
        x_47 = None
        x_49 = torch._C._nn.linear(
            x_48,
            l_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_48 = l_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks1_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_50 = torch.nn.functional.dropout(x_49, 0.0, False, False)
        x_49 = None
        x_51 = x_44 + x_50
        x_44 = x_50 = None
        getitem_38 = x_51[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        x_51 = None
        reshape_12 = getitem_38.reshape(1, 56, 56, -1)
        getitem_38 = None
        permute_3 = reshape_12.permute(0, 3, 1, 2)
        reshape_12 = None
        x1_nocls = permute_3.contiguous()
        permute_3 = None
        x_52 = torch.conv2d(
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
        flatten_4 = x_52.flatten(2)
        x_52 = None
        x_53 = flatten_4.transpose(1, 2)
        flatten_4 = None
        x_54 = torch.nn.functional.layer_norm(
            x_53,
            (128,),
            l_self_modules_patch_embed2_modules_norm_parameters_weight_,
            l_self_modules_patch_embed2_modules_norm_parameters_bias_,
            1e-05,
        )
        x_53 = (
            l_self_modules_patch_embed2_modules_norm_parameters_weight_
        ) = l_self_modules_patch_embed2_modules_norm_parameters_bias_ = None
        cls_tokens_1 = l_self_parameters_cls_token2_.expand(1, -1, -1)
        l_self_parameters_cls_token2_ = None
        x_55 = torch.cat((cls_tokens_1, x_54), dim=1)
        cls_tokens_1 = x_54 = None
        cls_token_3 = x_55[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_3 = x_55[(slice(None, None, None), slice(1, None, None))]
        x_55 = None
        transpose_20 = img_tokens_3.transpose(1, 2)
        img_tokens_3 = None
        feat_3 = transpose_20.view(1, 128, 28, 28)
        transpose_20 = None
        conv2d_14 = torch.conv2d(
            feat_3,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
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
            (128,),
            l_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear_12 = torch._C._nn.linear(
            x_59,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_59 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_13 = linear_12.reshape(1, 785, 3, 8, 16)
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
        v_img_7 = transpose_23.reshape(1, 128, 28, 28)
        transpose_23 = None
        split_3 = torch.functional.split(v_img_7, [32, 48, 48], dim=1)
        v_img_7 = None
        getitem_46 = split_3[0]
        getitem_47 = split_3[1]
        getitem_48 = split_3[2]
        split_3 = None
        conv2d_15 = torch.conv2d(
            getitem_46,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        getitem_46 = None
        conv2d_16 = torch.conv2d(
            getitem_47,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            48,
        )
        getitem_47 = None
        conv2d_17 = torch.conv2d(
            getitem_48,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            48,
        )
        getitem_48 = None
        conv_v_img_6 = torch.cat([conv2d_15, conv2d_16, conv2d_17], dim=1)
        conv2d_15 = conv2d_16 = conv2d_17 = None
        reshape_15 = conv_v_img_6.reshape(1, 8, 16, 784)
        conv_v_img_6 = None
        conv_v_img_7 = reshape_15.transpose(-1, -2)
        reshape_15 = None
        EV_hat_6 = q_img_3 * conv_v_img_7
        q_img_3 = conv_v_img_7 = None
        EV_hat_7 = torch._C._nn.pad(EV_hat_6, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_6 = None
        mul_7 = 0.25 * factor_att_7
        factor_att_7 = None
        x_60 = mul_7 + EV_hat_7
        mul_7 = EV_hat_7 = None
        transpose_25 = x_60.transpose(1, 2)
        x_60 = None
        x_61 = transpose_25.reshape(1, 785, 128)
        transpose_25 = None
        x_62 = torch._C._nn.linear(
            x_61,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_61 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_63 = torch.nn.functional.dropout(x_62, 0.0, False, False)
        x_62 = None
        x_64 = x_58 + x_63
        x_58 = x_63 = None
        x_65 = torch.nn.functional.layer_norm(
            x_64,
            (128,),
            l_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_66 = torch._C._nn.linear(
            x_65,
            l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_65 = l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_67 = torch._C._nn.gelu(x_66, approximate="none")
        x_66 = None
        x_68 = torch.nn.functional.dropout(x_67, 0.0, False, False)
        x_67 = None
        x_69 = torch._C._nn.linear(
            x_68,
            l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_68 = l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_70 = torch.nn.functional.dropout(x_69, 0.0, False, False)
        x_69 = None
        x_71 = x_64 + x_70
        x_64 = x_70 = None
        cls_token_4 = x_71[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_4 = x_71[(slice(None, None, None), slice(1, None, None))]
        x_71 = None
        transpose_26 = img_tokens_4.transpose(1, 2)
        img_tokens_4 = None
        feat_4 = transpose_26.view(1, 128, 28, 28)
        transpose_26 = None
        conv2d_18 = torch.conv2d(
            feat_4,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        x_72 = conv2d_18 + feat_4
        conv2d_18 = feat_4 = None
        flatten_6 = x_72.flatten(2)
        x_72 = None
        x_73 = flatten_6.transpose(1, 2)
        flatten_6 = None
        x_74 = torch.cat((cls_token_4, x_73), dim=1)
        cls_token_4 = x_73 = None
        x_75 = torch.nn.functional.layer_norm(
            x_74,
            (128,),
            l_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_1_modules_norm1_parameters_bias_
        ) = None
        linear_16 = torch._C._nn.linear(
            x_75,
            l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_75 = l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_17 = linear_16.reshape(1, 785, 3, 8, 16)
        linear_16 = None
        qkv_4 = reshape_17.permute(2, 0, 3, 1, 4)
        reshape_17 = None
        unbind_4 = qkv_4.unbind(0)
        qkv_4 = None
        q_4 = unbind_4[0]
        k_4 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        k_softmax_4 = k_4.softmax(dim=2)
        k_4 = None
        transpose_28 = k_softmax_4.transpose(-1, -2)
        k_softmax_4 = None
        factor_att_8 = transpose_28 @ v_4
        transpose_28 = None
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
        transpose_29 = v_img_8.transpose(-1, -2)
        v_img_8 = None
        v_img_9 = transpose_29.reshape(1, 128, 28, 28)
        transpose_29 = None
        split_4 = torch.functional.split(v_img_9, [32, 48, 48], dim=1)
        v_img_9 = None
        getitem_56 = split_4[0]
        getitem_57 = split_4[1]
        getitem_58 = split_4[2]
        split_4 = None
        conv2d_19 = torch.conv2d(
            getitem_56,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        getitem_56 = None
        conv2d_20 = torch.conv2d(
            getitem_57,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            48,
        )
        getitem_57 = None
        conv2d_21 = torch.conv2d(
            getitem_58,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            48,
        )
        getitem_58 = None
        conv_v_img_8 = torch.cat([conv2d_19, conv2d_20, conv2d_21], dim=1)
        conv2d_19 = conv2d_20 = conv2d_21 = None
        reshape_19 = conv_v_img_8.reshape(1, 8, 16, 784)
        conv_v_img_8 = None
        conv_v_img_9 = reshape_19.transpose(-1, -2)
        reshape_19 = None
        EV_hat_8 = q_img_4 * conv_v_img_9
        q_img_4 = conv_v_img_9 = None
        EV_hat_9 = torch._C._nn.pad(EV_hat_8, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_8 = None
        mul_9 = 0.25 * factor_att_9
        factor_att_9 = None
        x_76 = mul_9 + EV_hat_9
        mul_9 = EV_hat_9 = None
        transpose_31 = x_76.transpose(1, 2)
        x_76 = None
        x_77 = transpose_31.reshape(1, 785, 128)
        transpose_31 = None
        x_78 = torch._C._nn.linear(
            x_77,
            l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_77 = l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks2_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_79 = torch.nn.functional.dropout(x_78, 0.0, False, False)
        x_78 = None
        x_80 = x_74 + x_79
        x_74 = x_79 = None
        x_81 = torch.nn.functional.layer_norm(
            x_80,
            (128,),
            l_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_82 = torch._C._nn.linear(
            x_81,
            l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_81 = l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_83 = torch._C._nn.gelu(x_82, approximate="none")
        x_82 = None
        x_84 = torch.nn.functional.dropout(x_83, 0.0, False, False)
        x_83 = None
        x_85 = torch._C._nn.linear(
            x_84,
            l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_84 = l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks2_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_86 = torch.nn.functional.dropout(x_85, 0.0, False, False)
        x_85 = None
        x_87 = x_80 + x_86
        x_80 = x_86 = None
        cls_token_5 = x_87[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_5 = x_87[(slice(None, None, None), slice(1, None, None))]
        x_87 = None
        transpose_32 = img_tokens_5.transpose(1, 2)
        img_tokens_5 = None
        feat_5 = transpose_32.view(1, 128, 28, 28)
        transpose_32 = None
        conv2d_22 = torch.conv2d(
            feat_5,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        x_88 = conv2d_22 + feat_5
        conv2d_22 = feat_5 = None
        flatten_7 = x_88.flatten(2)
        x_88 = None
        x_89 = flatten_7.transpose(1, 2)
        flatten_7 = None
        x_90 = torch.cat((cls_token_5, x_89), dim=1)
        cls_token_5 = x_89 = None
        x_91 = torch.nn.functional.layer_norm(
            x_90,
            (128,),
            l_self_modules_serial_blocks2_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_2_modules_norm1_parameters_bias_
        ) = None
        linear_20 = torch._C._nn.linear(
            x_91,
            l_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_91 = l_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_21 = linear_20.reshape(1, 785, 3, 8, 16)
        linear_20 = None
        qkv_5 = reshape_21.permute(2, 0, 3, 1, 4)
        reshape_21 = None
        unbind_5 = qkv_5.unbind(0)
        qkv_5 = None
        q_5 = unbind_5[0]
        k_5 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        k_softmax_5 = k_5.softmax(dim=2)
        k_5 = None
        transpose_34 = k_softmax_5.transpose(-1, -2)
        k_softmax_5 = None
        factor_att_10 = transpose_34 @ v_5
        transpose_34 = None
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
        transpose_35 = v_img_10.transpose(-1, -2)
        v_img_10 = None
        v_img_11 = transpose_35.reshape(1, 128, 28, 28)
        transpose_35 = None
        split_5 = torch.functional.split(v_img_11, [32, 48, 48], dim=1)
        v_img_11 = None
        getitem_66 = split_5[0]
        getitem_67 = split_5[1]
        getitem_68 = split_5[2]
        split_5 = None
        conv2d_23 = torch.conv2d(
            getitem_66,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        getitem_66 = None
        conv2d_24 = torch.conv2d(
            getitem_67,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            48,
        )
        getitem_67 = None
        conv2d_25 = torch.conv2d(
            getitem_68,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            48,
        )
        getitem_68 = None
        conv_v_img_10 = torch.cat([conv2d_23, conv2d_24, conv2d_25], dim=1)
        conv2d_23 = conv2d_24 = conv2d_25 = None
        reshape_23 = conv_v_img_10.reshape(1, 8, 16, 784)
        conv_v_img_10 = None
        conv_v_img_11 = reshape_23.transpose(-1, -2)
        reshape_23 = None
        EV_hat_10 = q_img_5 * conv_v_img_11
        q_img_5 = conv_v_img_11 = None
        EV_hat_11 = torch._C._nn.pad(EV_hat_10, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_10 = None
        mul_11 = 0.25 * factor_att_11
        factor_att_11 = None
        x_92 = mul_11 + EV_hat_11
        mul_11 = EV_hat_11 = None
        transpose_37 = x_92.transpose(1, 2)
        x_92 = None
        x_93 = transpose_37.reshape(1, 785, 128)
        transpose_37 = None
        x_94 = torch._C._nn.linear(
            x_93,
            l_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_93 = l_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks2_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_95 = torch.nn.functional.dropout(x_94, 0.0, False, False)
        x_94 = None
        x_96 = x_90 + x_95
        x_90 = x_95 = None
        x_97 = torch.nn.functional.layer_norm(
            x_96,
            (128,),
            l_self_modules_serial_blocks2_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_98 = torch._C._nn.linear(
            x_97,
            l_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_97 = l_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_99 = torch._C._nn.gelu(x_98, approximate="none")
        x_98 = None
        x_100 = torch.nn.functional.dropout(x_99, 0.0, False, False)
        x_99 = None
        x_101 = torch._C._nn.linear(
            x_100,
            l_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_100 = l_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks2_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_102 = torch.nn.functional.dropout(x_101, 0.0, False, False)
        x_101 = None
        x_103 = x_96 + x_102
        x_96 = x_102 = None
        cls_token_6 = x_103[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_6 = x_103[(slice(None, None, None), slice(1, None, None))]
        x_103 = None
        transpose_38 = img_tokens_6.transpose(1, 2)
        img_tokens_6 = None
        feat_6 = transpose_38.view(1, 128, 28, 28)
        transpose_38 = None
        conv2d_26 = torch.conv2d(
            feat_6,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_ = (None)
        x_104 = conv2d_26 + feat_6
        conv2d_26 = feat_6 = None
        flatten_8 = x_104.flatten(2)
        x_104 = None
        x_105 = flatten_8.transpose(1, 2)
        flatten_8 = None
        x_106 = torch.cat((cls_token_6, x_105), dim=1)
        cls_token_6 = x_105 = None
        x_107 = torch.nn.functional.layer_norm(
            x_106,
            (128,),
            l_self_modules_serial_blocks2_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_3_modules_norm1_parameters_bias_
        ) = None
        linear_24 = torch._C._nn.linear(
            x_107,
            l_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_107 = l_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_25 = linear_24.reshape(1, 785, 3, 8, 16)
        linear_24 = None
        qkv_6 = reshape_25.permute(2, 0, 3, 1, 4)
        reshape_25 = None
        unbind_6 = qkv_6.unbind(0)
        qkv_6 = None
        q_6 = unbind_6[0]
        k_6 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        k_softmax_6 = k_6.softmax(dim=2)
        k_6 = None
        transpose_40 = k_softmax_6.transpose(-1, -2)
        k_softmax_6 = None
        factor_att_12 = transpose_40 @ v_6
        transpose_40 = None
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
        transpose_41 = v_img_12.transpose(-1, -2)
        v_img_12 = None
        v_img_13 = transpose_41.reshape(1, 128, 28, 28)
        transpose_41 = None
        split_6 = torch.functional.split(v_img_13, [32, 48, 48], dim=1)
        v_img_13 = None
        getitem_76 = split_6[0]
        getitem_77 = split_6[1]
        getitem_78 = split_6[2]
        split_6 = None
        conv2d_27 = torch.conv2d(
            getitem_76,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        getitem_76 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_28 = torch.conv2d(
            getitem_77,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            48,
        )
        getitem_77 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_29 = torch.conv2d(
            getitem_78,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            48,
        )
        getitem_78 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
        conv_v_img_12 = torch.cat([conv2d_27, conv2d_28, conv2d_29], dim=1)
        conv2d_27 = conv2d_28 = conv2d_29 = None
        reshape_27 = conv_v_img_12.reshape(1, 8, 16, 784)
        conv_v_img_12 = None
        conv_v_img_13 = reshape_27.transpose(-1, -2)
        reshape_27 = None
        EV_hat_12 = q_img_6 * conv_v_img_13
        q_img_6 = conv_v_img_13 = None
        EV_hat_13 = torch._C._nn.pad(EV_hat_12, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_12 = None
        mul_13 = 0.25 * factor_att_13
        factor_att_13 = None
        x_108 = mul_13 + EV_hat_13
        mul_13 = EV_hat_13 = None
        transpose_43 = x_108.transpose(1, 2)
        x_108 = None
        x_109 = transpose_43.reshape(1, 785, 128)
        transpose_43 = None
        x_110 = torch._C._nn.linear(
            x_109,
            l_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_109 = l_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks2_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_111 = torch.nn.functional.dropout(x_110, 0.0, False, False)
        x_110 = None
        x_112 = x_106 + x_111
        x_106 = x_111 = None
        x_113 = torch.nn.functional.layer_norm(
            x_112,
            (128,),
            l_self_modules_serial_blocks2_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_114 = torch._C._nn.linear(
            x_113,
            l_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_113 = l_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_115 = torch._C._nn.gelu(x_114, approximate="none")
        x_114 = None
        x_116 = torch.nn.functional.dropout(x_115, 0.0, False, False)
        x_115 = None
        x_117 = torch._C._nn.linear(
            x_116,
            l_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_116 = l_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks2_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_118 = torch.nn.functional.dropout(x_117, 0.0, False, False)
        x_117 = None
        x_119 = x_112 + x_118
        x_112 = x_118 = None
        getitem_79 = x_119[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        x_119 = None
        reshape_29 = getitem_79.reshape(1, 28, 28, -1)
        getitem_79 = None
        permute_8 = reshape_29.permute(0, 3, 1, 2)
        reshape_29 = None
        x2_nocls = permute_8.contiguous()
        permute_8 = None
        x_120 = torch.conv2d(
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
        flatten_9 = x_120.flatten(2)
        x_120 = None
        x_121 = flatten_9.transpose(1, 2)
        flatten_9 = None
        x_122 = torch.nn.functional.layer_norm(
            x_121,
            (320,),
            l_self_modules_patch_embed3_modules_norm_parameters_weight_,
            l_self_modules_patch_embed3_modules_norm_parameters_bias_,
            1e-05,
        )
        x_121 = (
            l_self_modules_patch_embed3_modules_norm_parameters_weight_
        ) = l_self_modules_patch_embed3_modules_norm_parameters_bias_ = None
        cls_tokens_2 = l_self_parameters_cls_token3_.expand(1, -1, -1)
        l_self_parameters_cls_token3_ = None
        x_123 = torch.cat((cls_tokens_2, x_122), dim=1)
        cls_tokens_2 = x_122 = None
        cls_token_7 = x_123[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_7 = x_123[(slice(None, None, None), slice(1, None, None))]
        x_123 = None
        transpose_45 = img_tokens_7.transpose(1, 2)
        img_tokens_7 = None
        feat_7 = transpose_45.view(1, 320, 14, 14)
        transpose_45 = None
        conv2d_31 = torch.conv2d(
            feat_7,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        x_124 = conv2d_31 + feat_7
        conv2d_31 = feat_7 = None
        flatten_10 = x_124.flatten(2)
        x_124 = None
        x_125 = flatten_10.transpose(1, 2)
        flatten_10 = None
        x_126 = torch.cat((cls_token_7, x_125), dim=1)
        cls_token_7 = x_125 = None
        x_127 = torch.nn.functional.layer_norm(
            x_126,
            (320,),
            l_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear_28 = torch._C._nn.linear(
            x_127,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_127 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_30 = linear_28.reshape(1, 197, 3, 8, 40)
        linear_28 = None
        qkv_7 = reshape_30.permute(2, 0, 3, 1, 4)
        reshape_30 = None
        unbind_7 = qkv_7.unbind(0)
        qkv_7 = None
        q_7 = unbind_7[0]
        k_7 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        k_softmax_7 = k_7.softmax(dim=2)
        k_7 = None
        transpose_47 = k_softmax_7.transpose(-1, -2)
        k_softmax_7 = None
        factor_att_14 = transpose_47 @ v_7
        transpose_47 = None
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
        transpose_48 = v_img_14.transpose(-1, -2)
        v_img_14 = None
        v_img_15 = transpose_48.reshape(1, 320, 14, 14)
        transpose_48 = None
        split_7 = torch.functional.split(v_img_15, [80, 120, 120], dim=1)
        v_img_15 = None
        getitem_87 = split_7[0]
        getitem_88 = split_7[1]
        getitem_89 = split_7[2]
        split_7 = None
        conv2d_32 = torch.conv2d(
            getitem_87,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        getitem_87 = None
        conv2d_33 = torch.conv2d(
            getitem_88,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_88 = None
        conv2d_34 = torch.conv2d(
            getitem_89,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_89 = None
        conv_v_img_14 = torch.cat([conv2d_32, conv2d_33, conv2d_34], dim=1)
        conv2d_32 = conv2d_33 = conv2d_34 = None
        reshape_32 = conv_v_img_14.reshape(1, 8, 40, 196)
        conv_v_img_14 = None
        conv_v_img_15 = reshape_32.transpose(-1, -2)
        reshape_32 = None
        EV_hat_14 = q_img_7 * conv_v_img_15
        q_img_7 = conv_v_img_15 = None
        EV_hat_15 = torch._C._nn.pad(EV_hat_14, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_14 = None
        mul_15 = 0.15811388300841897 * factor_att_15
        factor_att_15 = None
        x_128 = mul_15 + EV_hat_15
        mul_15 = EV_hat_15 = None
        transpose_50 = x_128.transpose(1, 2)
        x_128 = None
        x_129 = transpose_50.reshape(1, 197, 320)
        transpose_50 = None
        x_130 = torch._C._nn.linear(
            x_129,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_129 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_131 = torch.nn.functional.dropout(x_130, 0.0, False, False)
        x_130 = None
        x_132 = x_126 + x_131
        x_126 = x_131 = None
        x_133 = torch.nn.functional.layer_norm(
            x_132,
            (320,),
            l_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_134 = torch._C._nn.linear(
            x_133,
            l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_133 = l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_135 = torch._C._nn.gelu(x_134, approximate="none")
        x_134 = None
        x_136 = torch.nn.functional.dropout(x_135, 0.0, False, False)
        x_135 = None
        x_137 = torch._C._nn.linear(
            x_136,
            l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_136 = l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_138 = torch.nn.functional.dropout(x_137, 0.0, False, False)
        x_137 = None
        x_139 = x_132 + x_138
        x_132 = x_138 = None
        cls_token_8 = x_139[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_8 = x_139[(slice(None, None, None), slice(1, None, None))]
        x_139 = None
        transpose_51 = img_tokens_8.transpose(1, 2)
        img_tokens_8 = None
        feat_8 = transpose_51.view(1, 320, 14, 14)
        transpose_51 = None
        conv2d_35 = torch.conv2d(
            feat_8,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        x_140 = conv2d_35 + feat_8
        conv2d_35 = feat_8 = None
        flatten_11 = x_140.flatten(2)
        x_140 = None
        x_141 = flatten_11.transpose(1, 2)
        flatten_11 = None
        x_142 = torch.cat((cls_token_8, x_141), dim=1)
        cls_token_8 = x_141 = None
        x_143 = torch.nn.functional.layer_norm(
            x_142,
            (320,),
            l_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_bias_
        ) = None
        linear_32 = torch._C._nn.linear(
            x_143,
            l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_143 = l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_34 = linear_32.reshape(1, 197, 3, 8, 40)
        linear_32 = None
        qkv_8 = reshape_34.permute(2, 0, 3, 1, 4)
        reshape_34 = None
        unbind_8 = qkv_8.unbind(0)
        qkv_8 = None
        q_8 = unbind_8[0]
        k_8 = unbind_8[1]
        v_8 = unbind_8[2]
        unbind_8 = None
        k_softmax_8 = k_8.softmax(dim=2)
        k_8 = None
        transpose_53 = k_softmax_8.transpose(-1, -2)
        k_softmax_8 = None
        factor_att_16 = transpose_53 @ v_8
        transpose_53 = None
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
        transpose_54 = v_img_16.transpose(-1, -2)
        v_img_16 = None
        v_img_17 = transpose_54.reshape(1, 320, 14, 14)
        transpose_54 = None
        split_8 = torch.functional.split(v_img_17, [80, 120, 120], dim=1)
        v_img_17 = None
        getitem_97 = split_8[0]
        getitem_98 = split_8[1]
        getitem_99 = split_8[2]
        split_8 = None
        conv2d_36 = torch.conv2d(
            getitem_97,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        getitem_97 = None
        conv2d_37 = torch.conv2d(
            getitem_98,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_98 = None
        conv2d_38 = torch.conv2d(
            getitem_99,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_99 = None
        conv_v_img_16 = torch.cat([conv2d_36, conv2d_37, conv2d_38], dim=1)
        conv2d_36 = conv2d_37 = conv2d_38 = None
        reshape_36 = conv_v_img_16.reshape(1, 8, 40, 196)
        conv_v_img_16 = None
        conv_v_img_17 = reshape_36.transpose(-1, -2)
        reshape_36 = None
        EV_hat_16 = q_img_8 * conv_v_img_17
        q_img_8 = conv_v_img_17 = None
        EV_hat_17 = torch._C._nn.pad(EV_hat_16, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_16 = None
        mul_17 = 0.15811388300841897 * factor_att_17
        factor_att_17 = None
        x_144 = mul_17 + EV_hat_17
        mul_17 = EV_hat_17 = None
        transpose_56 = x_144.transpose(1, 2)
        x_144 = None
        x_145 = transpose_56.reshape(1, 197, 320)
        transpose_56 = None
        x_146 = torch._C._nn.linear(
            x_145,
            l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_145 = l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_147 = torch.nn.functional.dropout(x_146, 0.0, False, False)
        x_146 = None
        x_148 = x_142 + x_147
        x_142 = x_147 = None
        x_149 = torch.nn.functional.layer_norm(
            x_148,
            (320,),
            l_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_150 = torch._C._nn.linear(
            x_149,
            l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_149 = l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_151 = torch._C._nn.gelu(x_150, approximate="none")
        x_150 = None
        x_152 = torch.nn.functional.dropout(x_151, 0.0, False, False)
        x_151 = None
        x_153 = torch._C._nn.linear(
            x_152,
            l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_152 = l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_154 = torch.nn.functional.dropout(x_153, 0.0, False, False)
        x_153 = None
        x_155 = x_148 + x_154
        x_148 = x_154 = None
        cls_token_9 = x_155[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_9 = x_155[(slice(None, None, None), slice(1, None, None))]
        x_155 = None
        transpose_57 = img_tokens_9.transpose(1, 2)
        img_tokens_9 = None
        feat_9 = transpose_57.view(1, 320, 14, 14)
        transpose_57 = None
        conv2d_39 = torch.conv2d(
            feat_9,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        x_156 = conv2d_39 + feat_9
        conv2d_39 = feat_9 = None
        flatten_12 = x_156.flatten(2)
        x_156 = None
        x_157 = flatten_12.transpose(1, 2)
        flatten_12 = None
        x_158 = torch.cat((cls_token_9, x_157), dim=1)
        cls_token_9 = x_157 = None
        x_159 = torch.nn.functional.layer_norm(
            x_158,
            (320,),
            l_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_bias_
        ) = None
        linear_36 = torch._C._nn.linear(
            x_159,
            l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_159 = l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_38 = linear_36.reshape(1, 197, 3, 8, 40)
        linear_36 = None
        qkv_9 = reshape_38.permute(2, 0, 3, 1, 4)
        reshape_38 = None
        unbind_9 = qkv_9.unbind(0)
        qkv_9 = None
        q_9 = unbind_9[0]
        k_9 = unbind_9[1]
        v_9 = unbind_9[2]
        unbind_9 = None
        k_softmax_9 = k_9.softmax(dim=2)
        k_9 = None
        transpose_59 = k_softmax_9.transpose(-1, -2)
        k_softmax_9 = None
        factor_att_18 = transpose_59 @ v_9
        transpose_59 = None
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
        transpose_60 = v_img_18.transpose(-1, -2)
        v_img_18 = None
        v_img_19 = transpose_60.reshape(1, 320, 14, 14)
        transpose_60 = None
        split_9 = torch.functional.split(v_img_19, [80, 120, 120], dim=1)
        v_img_19 = None
        getitem_107 = split_9[0]
        getitem_108 = split_9[1]
        getitem_109 = split_9[2]
        split_9 = None
        conv2d_40 = torch.conv2d(
            getitem_107,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        getitem_107 = None
        conv2d_41 = torch.conv2d(
            getitem_108,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_108 = None
        conv2d_42 = torch.conv2d(
            getitem_109,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_109 = None
        conv_v_img_18 = torch.cat([conv2d_40, conv2d_41, conv2d_42], dim=1)
        conv2d_40 = conv2d_41 = conv2d_42 = None
        reshape_40 = conv_v_img_18.reshape(1, 8, 40, 196)
        conv_v_img_18 = None
        conv_v_img_19 = reshape_40.transpose(-1, -2)
        reshape_40 = None
        EV_hat_18 = q_img_9 * conv_v_img_19
        q_img_9 = conv_v_img_19 = None
        EV_hat_19 = torch._C._nn.pad(EV_hat_18, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_18 = None
        mul_19 = 0.15811388300841897 * factor_att_19
        factor_att_19 = None
        x_160 = mul_19 + EV_hat_19
        mul_19 = EV_hat_19 = None
        transpose_62 = x_160.transpose(1, 2)
        x_160 = None
        x_161 = transpose_62.reshape(1, 197, 320)
        transpose_62 = None
        x_162 = torch._C._nn.linear(
            x_161,
            l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_161 = l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_163 = torch.nn.functional.dropout(x_162, 0.0, False, False)
        x_162 = None
        x_164 = x_158 + x_163
        x_158 = x_163 = None
        x_165 = torch.nn.functional.layer_norm(
            x_164,
            (320,),
            l_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_166 = torch._C._nn.linear(
            x_165,
            l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_165 = l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_167 = torch._C._nn.gelu(x_166, approximate="none")
        x_166 = None
        x_168 = torch.nn.functional.dropout(x_167, 0.0, False, False)
        x_167 = None
        x_169 = torch._C._nn.linear(
            x_168,
            l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_168 = l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_170 = torch.nn.functional.dropout(x_169, 0.0, False, False)
        x_169 = None
        x_171 = x_164 + x_170
        x_164 = x_170 = None
        cls_token_10 = x_171[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_10 = x_171[(slice(None, None, None), slice(1, None, None))]
        x_171 = None
        transpose_63 = img_tokens_10.transpose(1, 2)
        img_tokens_10 = None
        feat_10 = transpose_63.view(1, 320, 14, 14)
        transpose_63 = None
        conv2d_43 = torch.conv2d(
            feat_10,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        x_172 = conv2d_43 + feat_10
        conv2d_43 = feat_10 = None
        flatten_13 = x_172.flatten(2)
        x_172 = None
        x_173 = flatten_13.transpose(1, 2)
        flatten_13 = None
        x_174 = torch.cat((cls_token_10, x_173), dim=1)
        cls_token_10 = x_173 = None
        x_175 = torch.nn.functional.layer_norm(
            x_174,
            (320,),
            l_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_bias_
        ) = None
        linear_40 = torch._C._nn.linear(
            x_175,
            l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_175 = l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_42 = linear_40.reshape(1, 197, 3, 8, 40)
        linear_40 = None
        qkv_10 = reshape_42.permute(2, 0, 3, 1, 4)
        reshape_42 = None
        unbind_10 = qkv_10.unbind(0)
        qkv_10 = None
        q_10 = unbind_10[0]
        k_10 = unbind_10[1]
        v_10 = unbind_10[2]
        unbind_10 = None
        k_softmax_10 = k_10.softmax(dim=2)
        k_10 = None
        transpose_65 = k_softmax_10.transpose(-1, -2)
        k_softmax_10 = None
        factor_att_20 = transpose_65 @ v_10
        transpose_65 = None
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
        transpose_66 = v_img_20.transpose(-1, -2)
        v_img_20 = None
        v_img_21 = transpose_66.reshape(1, 320, 14, 14)
        transpose_66 = None
        split_10 = torch.functional.split(v_img_21, [80, 120, 120], dim=1)
        v_img_21 = None
        getitem_117 = split_10[0]
        getitem_118 = split_10[1]
        getitem_119 = split_10[2]
        split_10 = None
        conv2d_44 = torch.conv2d(
            getitem_117,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        getitem_117 = None
        conv2d_45 = torch.conv2d(
            getitem_118,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_118 = None
        conv2d_46 = torch.conv2d(
            getitem_119,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_119 = None
        conv_v_img_20 = torch.cat([conv2d_44, conv2d_45, conv2d_46], dim=1)
        conv2d_44 = conv2d_45 = conv2d_46 = None
        reshape_44 = conv_v_img_20.reshape(1, 8, 40, 196)
        conv_v_img_20 = None
        conv_v_img_21 = reshape_44.transpose(-1, -2)
        reshape_44 = None
        EV_hat_20 = q_img_10 * conv_v_img_21
        q_img_10 = conv_v_img_21 = None
        EV_hat_21 = torch._C._nn.pad(EV_hat_20, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_20 = None
        mul_21 = 0.15811388300841897 * factor_att_21
        factor_att_21 = None
        x_176 = mul_21 + EV_hat_21
        mul_21 = EV_hat_21 = None
        transpose_68 = x_176.transpose(1, 2)
        x_176 = None
        x_177 = transpose_68.reshape(1, 197, 320)
        transpose_68 = None
        x_178 = torch._C._nn.linear(
            x_177,
            l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_177 = l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_179 = torch.nn.functional.dropout(x_178, 0.0, False, False)
        x_178 = None
        x_180 = x_174 + x_179
        x_174 = x_179 = None
        x_181 = torch.nn.functional.layer_norm(
            x_180,
            (320,),
            l_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_182 = torch._C._nn.linear(
            x_181,
            l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_181 = l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_183 = torch._C._nn.gelu(x_182, approximate="none")
        x_182 = None
        x_184 = torch.nn.functional.dropout(x_183, 0.0, False, False)
        x_183 = None
        x_185 = torch._C._nn.linear(
            x_184,
            l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_184 = l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_186 = torch.nn.functional.dropout(x_185, 0.0, False, False)
        x_185 = None
        x_187 = x_180 + x_186
        x_180 = x_186 = None
        cls_token_11 = x_187[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_11 = x_187[(slice(None, None, None), slice(1, None, None))]
        x_187 = None
        transpose_69 = img_tokens_11.transpose(1, 2)
        img_tokens_11 = None
        feat_11 = transpose_69.view(1, 320, 14, 14)
        transpose_69 = None
        conv2d_47 = torch.conv2d(
            feat_11,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        x_188 = conv2d_47 + feat_11
        conv2d_47 = feat_11 = None
        flatten_14 = x_188.flatten(2)
        x_188 = None
        x_189 = flatten_14.transpose(1, 2)
        flatten_14 = None
        x_190 = torch.cat((cls_token_11, x_189), dim=1)
        cls_token_11 = x_189 = None
        x_191 = torch.nn.functional.layer_norm(
            x_190,
            (320,),
            l_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_bias_
        ) = None
        linear_44 = torch._C._nn.linear(
            x_191,
            l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_191 = l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_46 = linear_44.reshape(1, 197, 3, 8, 40)
        linear_44 = None
        qkv_11 = reshape_46.permute(2, 0, 3, 1, 4)
        reshape_46 = None
        unbind_11 = qkv_11.unbind(0)
        qkv_11 = None
        q_11 = unbind_11[0]
        k_11 = unbind_11[1]
        v_11 = unbind_11[2]
        unbind_11 = None
        k_softmax_11 = k_11.softmax(dim=2)
        k_11 = None
        transpose_71 = k_softmax_11.transpose(-1, -2)
        k_softmax_11 = None
        factor_att_22 = transpose_71 @ v_11
        transpose_71 = None
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
        transpose_72 = v_img_22.transpose(-1, -2)
        v_img_22 = None
        v_img_23 = transpose_72.reshape(1, 320, 14, 14)
        transpose_72 = None
        split_11 = torch.functional.split(v_img_23, [80, 120, 120], dim=1)
        v_img_23 = None
        getitem_127 = split_11[0]
        getitem_128 = split_11[1]
        getitem_129 = split_11[2]
        split_11 = None
        conv2d_48 = torch.conv2d(
            getitem_127,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        getitem_127 = None
        conv2d_49 = torch.conv2d(
            getitem_128,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_128 = None
        conv2d_50 = torch.conv2d(
            getitem_129,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_129 = None
        conv_v_img_22 = torch.cat([conv2d_48, conv2d_49, conv2d_50], dim=1)
        conv2d_48 = conv2d_49 = conv2d_50 = None
        reshape_48 = conv_v_img_22.reshape(1, 8, 40, 196)
        conv_v_img_22 = None
        conv_v_img_23 = reshape_48.transpose(-1, -2)
        reshape_48 = None
        EV_hat_22 = q_img_11 * conv_v_img_23
        q_img_11 = conv_v_img_23 = None
        EV_hat_23 = torch._C._nn.pad(EV_hat_22, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_22 = None
        mul_23 = 0.15811388300841897 * factor_att_23
        factor_att_23 = None
        x_192 = mul_23 + EV_hat_23
        mul_23 = EV_hat_23 = None
        transpose_74 = x_192.transpose(1, 2)
        x_192 = None
        x_193 = transpose_74.reshape(1, 197, 320)
        transpose_74 = None
        x_194 = torch._C._nn.linear(
            x_193,
            l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_193 = l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_195 = torch.nn.functional.dropout(x_194, 0.0, False, False)
        x_194 = None
        x_196 = x_190 + x_195
        x_190 = x_195 = None
        x_197 = torch.nn.functional.layer_norm(
            x_196,
            (320,),
            l_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_198 = torch._C._nn.linear(
            x_197,
            l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_197 = l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_199 = torch._C._nn.gelu(x_198, approximate="none")
        x_198 = None
        x_200 = torch.nn.functional.dropout(x_199, 0.0, False, False)
        x_199 = None
        x_201 = torch._C._nn.linear(
            x_200,
            l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_200 = l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_202 = torch.nn.functional.dropout(x_201, 0.0, False, False)
        x_201 = None
        x_203 = x_196 + x_202
        x_196 = x_202 = None
        cls_token_12 = x_203[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_12 = x_203[(slice(None, None, None), slice(1, None, None))]
        x_203 = None
        transpose_75 = img_tokens_12.transpose(1, 2)
        img_tokens_12 = None
        feat_12 = transpose_75.view(1, 320, 14, 14)
        transpose_75 = None
        conv2d_51 = torch.conv2d(
            feat_12,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_ = (None)
        x_204 = conv2d_51 + feat_12
        conv2d_51 = feat_12 = None
        flatten_15 = x_204.flatten(2)
        x_204 = None
        x_205 = flatten_15.transpose(1, 2)
        flatten_15 = None
        x_206 = torch.cat((cls_token_12, x_205), dim=1)
        cls_token_12 = x_205 = None
        x_207 = torch.nn.functional.layer_norm(
            x_206,
            (320,),
            l_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_bias_
        ) = None
        linear_48 = torch._C._nn.linear(
            x_207,
            l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_207 = l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_50 = linear_48.reshape(1, 197, 3, 8, 40)
        linear_48 = None
        qkv_12 = reshape_50.permute(2, 0, 3, 1, 4)
        reshape_50 = None
        unbind_12 = qkv_12.unbind(0)
        qkv_12 = None
        q_12 = unbind_12[0]
        k_12 = unbind_12[1]
        v_12 = unbind_12[2]
        unbind_12 = None
        k_softmax_12 = k_12.softmax(dim=2)
        k_12 = None
        transpose_77 = k_softmax_12.transpose(-1, -2)
        k_softmax_12 = None
        factor_att_24 = transpose_77 @ v_12
        transpose_77 = None
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
        transpose_78 = v_img_24.transpose(-1, -2)
        v_img_24 = None
        v_img_25 = transpose_78.reshape(1, 320, 14, 14)
        transpose_78 = None
        split_12 = torch.functional.split(v_img_25, [80, 120, 120], dim=1)
        v_img_25 = None
        getitem_137 = split_12[0]
        getitem_138 = split_12[1]
        getitem_139 = split_12[2]
        split_12 = None
        conv2d_52 = torch.conv2d(
            getitem_137,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        getitem_137 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_53 = torch.conv2d(
            getitem_138,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_138 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_54 = torch.conv2d(
            getitem_139,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_139 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
        conv_v_img_24 = torch.cat([conv2d_52, conv2d_53, conv2d_54], dim=1)
        conv2d_52 = conv2d_53 = conv2d_54 = None
        reshape_52 = conv_v_img_24.reshape(1, 8, 40, 196)
        conv_v_img_24 = None
        conv_v_img_25 = reshape_52.transpose(-1, -2)
        reshape_52 = None
        EV_hat_24 = q_img_12 * conv_v_img_25
        q_img_12 = conv_v_img_25 = None
        EV_hat_25 = torch._C._nn.pad(EV_hat_24, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_24 = None
        mul_25 = 0.15811388300841897 * factor_att_25
        factor_att_25 = None
        x_208 = mul_25 + EV_hat_25
        mul_25 = EV_hat_25 = None
        transpose_80 = x_208.transpose(1, 2)
        x_208 = None
        x_209 = transpose_80.reshape(1, 197, 320)
        transpose_80 = None
        x_210 = torch._C._nn.linear(
            x_209,
            l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_209 = l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_211 = torch.nn.functional.dropout(x_210, 0.0, False, False)
        x_210 = None
        x_212 = x_206 + x_211
        x_206 = x_211 = None
        x_213 = torch.nn.functional.layer_norm(
            x_212,
            (320,),
            l_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_214 = torch._C._nn.linear(
            x_213,
            l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_213 = l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_215 = torch._C._nn.gelu(x_214, approximate="none")
        x_214 = None
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        x_217 = torch._C._nn.linear(
            x_216,
            l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_216 = l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_218 = torch.nn.functional.dropout(x_217, 0.0, False, False)
        x_217 = None
        x_219 = x_212 + x_218
        x_212 = x_218 = None
        getitem_140 = x_219[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        x_219 = None
        reshape_54 = getitem_140.reshape(1, 14, 14, -1)
        getitem_140 = None
        permute_15 = reshape_54.permute(0, 3, 1, 2)
        reshape_54 = None
        x3_nocls = permute_15.contiguous()
        permute_15 = None
        x_220 = torch.conv2d(
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
        flatten_16 = x_220.flatten(2)
        x_220 = None
        x_221 = flatten_16.transpose(1, 2)
        flatten_16 = None
        x_222 = torch.nn.functional.layer_norm(
            x_221,
            (512,),
            l_self_modules_patch_embed4_modules_norm_parameters_weight_,
            l_self_modules_patch_embed4_modules_norm_parameters_bias_,
            1e-05,
        )
        x_221 = (
            l_self_modules_patch_embed4_modules_norm_parameters_weight_
        ) = l_self_modules_patch_embed4_modules_norm_parameters_bias_ = None
        cls_tokens_3 = l_self_parameters_cls_token4_.expand(1, -1, -1)
        l_self_parameters_cls_token4_ = None
        x_223 = torch.cat((cls_tokens_3, x_222), dim=1)
        cls_tokens_3 = x_222 = None
        cls_token_13 = x_223[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_13 = x_223[(slice(None, None, None), slice(1, None, None))]
        x_223 = None
        transpose_82 = img_tokens_13.transpose(1, 2)
        img_tokens_13 = None
        feat_13 = transpose_82.view(1, 512, 7, 7)
        transpose_82 = None
        conv2d_56 = torch.conv2d(
            feat_13,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_224 = conv2d_56 + feat_13
        conv2d_56 = feat_13 = None
        flatten_17 = x_224.flatten(2)
        x_224 = None
        x_225 = flatten_17.transpose(1, 2)
        flatten_17 = None
        x_226 = torch.cat((cls_token_13, x_225), dim=1)
        cls_token_13 = x_225 = None
        x_227 = torch.nn.functional.layer_norm(
            x_226,
            (512,),
            l_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear_52 = torch._C._nn.linear(
            x_227,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_227 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_55 = linear_52.reshape(1, 50, 3, 8, 64)
        linear_52 = None
        qkv_13 = reshape_55.permute(2, 0, 3, 1, 4)
        reshape_55 = None
        unbind_13 = qkv_13.unbind(0)
        qkv_13 = None
        q_13 = unbind_13[0]
        k_13 = unbind_13[1]
        v_13 = unbind_13[2]
        unbind_13 = None
        k_softmax_13 = k_13.softmax(dim=2)
        k_13 = None
        transpose_84 = k_softmax_13.transpose(-1, -2)
        k_softmax_13 = None
        factor_att_26 = transpose_84 @ v_13
        transpose_84 = None
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
        transpose_85 = v_img_26.transpose(-1, -2)
        v_img_26 = None
        v_img_27 = transpose_85.reshape(1, 512, 7, 7)
        transpose_85 = None
        split_13 = torch.functional.split(v_img_27, [128, 192, 192], dim=1)
        v_img_27 = None
        getitem_148 = split_13[0]
        getitem_149 = split_13[1]
        getitem_150 = split_13[2]
        split_13 = None
        conv2d_57 = torch.conv2d(
            getitem_148,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        getitem_148 = None
        conv2d_58 = torch.conv2d(
            getitem_149,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        getitem_149 = None
        conv2d_59 = torch.conv2d(
            getitem_150,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        getitem_150 = None
        conv_v_img_26 = torch.cat([conv2d_57, conv2d_58, conv2d_59], dim=1)
        conv2d_57 = conv2d_58 = conv2d_59 = None
        reshape_57 = conv_v_img_26.reshape(1, 8, 64, 49)
        conv_v_img_26 = None
        conv_v_img_27 = reshape_57.transpose(-1, -2)
        reshape_57 = None
        EV_hat_26 = q_img_13 * conv_v_img_27
        q_img_13 = conv_v_img_27 = None
        EV_hat_27 = torch._C._nn.pad(EV_hat_26, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_26 = None
        mul_27 = 0.125 * factor_att_27
        factor_att_27 = None
        x_228 = mul_27 + EV_hat_27
        mul_27 = EV_hat_27 = None
        transpose_87 = x_228.transpose(1, 2)
        x_228 = None
        x_229 = transpose_87.reshape(1, 50, 512)
        transpose_87 = None
        x_230 = torch._C._nn.linear(
            x_229,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_229 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_231 = torch.nn.functional.dropout(x_230, 0.0, False, False)
        x_230 = None
        x_232 = x_226 + x_231
        x_226 = x_231 = None
        x_233 = torch.nn.functional.layer_norm(
            x_232,
            (512,),
            l_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_234 = torch._C._nn.linear(
            x_233,
            l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_233 = l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_235 = torch._C._nn.gelu(x_234, approximate="none")
        x_234 = None
        x_236 = torch.nn.functional.dropout(x_235, 0.0, False, False)
        x_235 = None
        x_237 = torch._C._nn.linear(
            x_236,
            l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_236 = l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_238 = torch.nn.functional.dropout(x_237, 0.0, False, False)
        x_237 = None
        x_239 = x_232 + x_238
        x_232 = x_238 = None
        cls_token_14 = x_239[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_14 = x_239[(slice(None, None, None), slice(1, None, None))]
        x_239 = None
        transpose_88 = img_tokens_14.transpose(1, 2)
        img_tokens_14 = None
        feat_14 = transpose_88.view(1, 512, 7, 7)
        transpose_88 = None
        conv2d_60 = torch.conv2d(
            feat_14,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_240 = conv2d_60 + feat_14
        conv2d_60 = feat_14 = None
        flatten_18 = x_240.flatten(2)
        x_240 = None
        x_241 = flatten_18.transpose(1, 2)
        flatten_18 = None
        x_242 = torch.cat((cls_token_14, x_241), dim=1)
        cls_token_14 = x_241 = None
        x_243 = torch.nn.functional.layer_norm(
            x_242,
            (512,),
            l_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_bias_
        ) = None
        linear_56 = torch._C._nn.linear(
            x_243,
            l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_243 = l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_59 = linear_56.reshape(1, 50, 3, 8, 64)
        linear_56 = None
        qkv_14 = reshape_59.permute(2, 0, 3, 1, 4)
        reshape_59 = None
        unbind_14 = qkv_14.unbind(0)
        qkv_14 = None
        q_14 = unbind_14[0]
        k_14 = unbind_14[1]
        v_14 = unbind_14[2]
        unbind_14 = None
        k_softmax_14 = k_14.softmax(dim=2)
        k_14 = None
        transpose_90 = k_softmax_14.transpose(-1, -2)
        k_softmax_14 = None
        factor_att_28 = transpose_90 @ v_14
        transpose_90 = None
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
        transpose_91 = v_img_28.transpose(-1, -2)
        v_img_28 = None
        v_img_29 = transpose_91.reshape(1, 512, 7, 7)
        transpose_91 = None
        split_14 = torch.functional.split(v_img_29, [128, 192, 192], dim=1)
        v_img_29 = None
        getitem_158 = split_14[0]
        getitem_159 = split_14[1]
        getitem_160 = split_14[2]
        split_14 = None
        conv2d_61 = torch.conv2d(
            getitem_158,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        getitem_158 = None
        conv2d_62 = torch.conv2d(
            getitem_159,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        getitem_159 = None
        conv2d_63 = torch.conv2d(
            getitem_160,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        getitem_160 = None
        conv_v_img_28 = torch.cat([conv2d_61, conv2d_62, conv2d_63], dim=1)
        conv2d_61 = conv2d_62 = conv2d_63 = None
        reshape_61 = conv_v_img_28.reshape(1, 8, 64, 49)
        conv_v_img_28 = None
        conv_v_img_29 = reshape_61.transpose(-1, -2)
        reshape_61 = None
        EV_hat_28 = q_img_14 * conv_v_img_29
        q_img_14 = conv_v_img_29 = None
        EV_hat_29 = torch._C._nn.pad(EV_hat_28, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_28 = None
        mul_29 = 0.125 * factor_att_29
        factor_att_29 = None
        x_244 = mul_29 + EV_hat_29
        mul_29 = EV_hat_29 = None
        transpose_93 = x_244.transpose(1, 2)
        x_244 = None
        x_245 = transpose_93.reshape(1, 50, 512)
        transpose_93 = None
        x_246 = torch._C._nn.linear(
            x_245,
            l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_245 = l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_247 = torch.nn.functional.dropout(x_246, 0.0, False, False)
        x_246 = None
        x_248 = x_242 + x_247
        x_242 = x_247 = None
        x_249 = torch.nn.functional.layer_norm(
            x_248,
            (512,),
            l_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_250 = torch._C._nn.linear(
            x_249,
            l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_249 = l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_251 = torch._C._nn.gelu(x_250, approximate="none")
        x_250 = None
        x_252 = torch.nn.functional.dropout(x_251, 0.0, False, False)
        x_251 = None
        x_253 = torch._C._nn.linear(
            x_252,
            l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_252 = l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_254 = torch.nn.functional.dropout(x_253, 0.0, False, False)
        x_253 = None
        x_255 = x_248 + x_254
        x_248 = x_254 = None
        cls_token_15 = x_255[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_15 = x_255[(slice(None, None, None), slice(1, None, None))]
        x_255 = None
        transpose_94 = img_tokens_15.transpose(1, 2)
        img_tokens_15 = None
        feat_15 = transpose_94.view(1, 512, 7, 7)
        transpose_94 = None
        conv2d_64 = torch.conv2d(
            feat_15,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_ = (None)
        x_256 = conv2d_64 + feat_15
        conv2d_64 = feat_15 = None
        flatten_19 = x_256.flatten(2)
        x_256 = None
        x_257 = flatten_19.transpose(1, 2)
        flatten_19 = None
        x_258 = torch.cat((cls_token_15, x_257), dim=1)
        cls_token_15 = x_257 = None
        x_259 = torch.nn.functional.layer_norm(
            x_258,
            (512,),
            l_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_bias_
        ) = None
        linear_60 = torch._C._nn.linear(
            x_259,
            l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_259 = l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_63 = linear_60.reshape(1, 50, 3, 8, 64)
        linear_60 = None
        qkv_15 = reshape_63.permute(2, 0, 3, 1, 4)
        reshape_63 = None
        unbind_15 = qkv_15.unbind(0)
        qkv_15 = None
        q_15 = unbind_15[0]
        k_15 = unbind_15[1]
        v_15 = unbind_15[2]
        unbind_15 = None
        k_softmax_15 = k_15.softmax(dim=2)
        k_15 = None
        transpose_96 = k_softmax_15.transpose(-1, -2)
        k_softmax_15 = None
        factor_att_30 = transpose_96 @ v_15
        transpose_96 = None
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
        transpose_97 = v_img_30.transpose(-1, -2)
        v_img_30 = None
        v_img_31 = transpose_97.reshape(1, 512, 7, 7)
        transpose_97 = None
        split_15 = torch.functional.split(v_img_31, [128, 192, 192], dim=1)
        v_img_31 = None
        getitem_168 = split_15[0]
        getitem_169 = split_15[1]
        getitem_170 = split_15[2]
        split_15 = None
        conv2d_65 = torch.conv2d(
            getitem_168,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        getitem_168 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_66 = torch.conv2d(
            getitem_169,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        getitem_169 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_67 = torch.conv2d(
            getitem_170,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        getitem_170 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
        conv_v_img_30 = torch.cat([conv2d_65, conv2d_66, conv2d_67], dim=1)
        conv2d_65 = conv2d_66 = conv2d_67 = None
        reshape_65 = conv_v_img_30.reshape(1, 8, 64, 49)
        conv_v_img_30 = None
        conv_v_img_31 = reshape_65.transpose(-1, -2)
        reshape_65 = None
        EV_hat_30 = q_img_15 * conv_v_img_31
        q_img_15 = conv_v_img_31 = None
        EV_hat_31 = torch._C._nn.pad(EV_hat_30, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_30 = None
        mul_31 = 0.125 * factor_att_31
        factor_att_31 = None
        x_260 = mul_31 + EV_hat_31
        mul_31 = EV_hat_31 = None
        transpose_99 = x_260.transpose(1, 2)
        x_260 = None
        x_261 = transpose_99.reshape(1, 50, 512)
        transpose_99 = None
        x_262 = torch._C._nn.linear(
            x_261,
            l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_261 = l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_263 = torch.nn.functional.dropout(x_262, 0.0, False, False)
        x_262 = None
        x_264 = x_258 + x_263
        x_258 = x_263 = None
        x_265 = torch.nn.functional.layer_norm(
            x_264,
            (512,),
            l_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_266 = torch._C._nn.linear(
            x_265,
            l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_265 = l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_267 = torch._C._nn.gelu(x_266, approximate="none")
        x_266 = None
        x_268 = torch.nn.functional.dropout(x_267, 0.0, False, False)
        x_267 = None
        x_269 = torch._C._nn.linear(
            x_268,
            l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_268 = l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_270 = torch.nn.functional.dropout(x_269, 0.0, False, False)
        x_269 = None
        x_271 = x_264 + x_270
        x_264 = x_270 = None
        getitem_171 = x_271[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        reshape_67 = getitem_171.reshape(1, 7, 7, -1)
        getitem_171 = None
        permute_19 = reshape_67.permute(0, 3, 1, 2)
        reshape_67 = None
        x4_nocls = permute_19.contiguous()
        permute_19 = x4_nocls = None
        x_272 = torch.nn.functional.layer_norm(
            x_271,
            (512,),
            l_self_modules_norm4_parameters_weight_,
            l_self_modules_norm4_parameters_bias_,
            1e-06,
        )
        x_271 = (
            l_self_modules_norm4_parameters_weight_
        ) = l_self_modules_norm4_parameters_bias_ = None
        x_273 = x_272[(slice(None, None, None), 0)]
        x_272 = None
        x_274 = torch.nn.functional.dropout(x_273, 0.0, False, False)
        x_273 = None
        x_275 = torch._C._nn.linear(
            x_274,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_274 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_275,)
