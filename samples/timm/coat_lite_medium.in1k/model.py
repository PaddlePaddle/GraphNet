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
        L_self_modules_serial_blocks2_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_serial_blocks3_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_serial_blocks4_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_serial_blocks2_modules_4_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks2_modules_4_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks2_modules_4_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks2_modules_4_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks2_modules_4_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks2_modules_4_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks2_modules_4_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks2_modules_4_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks2_modules_5_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks2_modules_5_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks2_modules_5_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks2_modules_5_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks2_modules_5_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks2_modules_5_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks2_modules_5_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks2_modules_5_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_serial_blocks3_modules_6_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_6_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_6_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_6_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks3_modules_6_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_6_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_6_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_6_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks3_modules_7_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_7_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_7_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_7_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks3_modules_7_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_7_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_7_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_7_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks3_modules_8_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_8_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_8_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_8_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks3_modules_8_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_8_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_8_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_8_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks3_modules_9_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_9_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_9_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_9_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks3_modules_9_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks3_modules_9_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks3_modules_9_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks3_modules_9_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_serial_blocks4_modules_3_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_3_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_3_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_3_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks4_modules_3_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_3_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_3_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_3_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks4_modules_4_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_4_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_4_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_4_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks4_modules_4_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_4_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_4_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_4_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks4_modules_5_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_5_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_5_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_5_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks4_modules_5_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_5_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_5_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_5_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks4_modules_6_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_6_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_6_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_6_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks4_modules_6_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_6_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_6_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_6_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_serial_blocks4_modules_7_modules_norm1_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_7_modules_norm1_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_7_modules_norm1_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_7_modules_norm1_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_qkv_parameters_weight_ = L_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_qkv_parameters_weight_
        l_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_qkv_parameters_bias_ = L_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_qkv_parameters_bias_
        l_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_proj_parameters_weight_ = L_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_proj_parameters_weight_
        l_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_proj_parameters_bias_ = L_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_proj_parameters_bias_
        l_self_modules_serial_blocks4_modules_7_modules_norm2_parameters_weight_ = (
            L_self_modules_serial_blocks4_modules_7_modules_norm2_parameters_weight_
        )
        l_self_modules_serial_blocks4_modules_7_modules_norm2_parameters_bias_ = (
            L_self_modules_serial_blocks4_modules_7_modules_norm2_parameters_bias_
        )
        l_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc2_parameters_bias_
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
            (128,),
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
        feat = transpose_1.view(1, 128, 56, 56)
        transpose_1 = None
        conv2d_1 = torch.conv2d(
            feat,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
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
            (128,),
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
        reshape = linear.reshape(1, 3137, 3, 8, 16)
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
        v_img_1 = transpose_4.reshape(1, 128, 56, 56)
        transpose_4 = None
        split = torch.functional.split(v_img_1, [32, 48, 48], dim=1)
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
            32,
        )
        getitem_15 = None
        conv2d_3 = torch.conv2d(
            getitem_16,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            48,
        )
        getitem_16 = None
        conv2d_4 = torch.conv2d(
            getitem_17,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            48,
        )
        getitem_17 = None
        conv_v_img = torch.cat([conv2d_2, conv2d_3, conv2d_4], dim=1)
        conv2d_2 = conv2d_3 = conv2d_4 = None
        reshape_2 = conv_v_img.reshape(1, 8, 16, 3136)
        conv_v_img = None
        conv_v_img_1 = reshape_2.transpose(-1, -2)
        reshape_2 = None
        EV_hat = q_img * conv_v_img_1
        q_img = conv_v_img_1 = None
        EV_hat_1 = torch._C._nn.pad(EV_hat, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat = None
        mul_1 = 0.25 * factor_att_1
        factor_att_1 = None
        x_8 = mul_1 + EV_hat_1
        mul_1 = EV_hat_1 = None
        transpose_6 = x_8.transpose(1, 2)
        x_8 = None
        x_9 = transpose_6.reshape(1, 3137, 128)
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
            (128,),
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
        feat_1 = transpose_7.view(1, 128, 56, 56)
        transpose_7 = None
        conv2d_5 = torch.conv2d(
            feat_1,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
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
            (128,),
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
        reshape_4 = linear_4.reshape(1, 3137, 3, 8, 16)
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
        v_img_3 = transpose_10.reshape(1, 128, 56, 56)
        transpose_10 = None
        split_1 = torch.functional.split(v_img_3, [32, 48, 48], dim=1)
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
            32,
        )
        getitem_25 = None
        conv2d_7 = torch.conv2d(
            getitem_26,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            48,
        )
        getitem_26 = None
        conv2d_8 = torch.conv2d(
            getitem_27,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            48,
        )
        getitem_27 = None
        conv_v_img_2 = torch.cat([conv2d_6, conv2d_7, conv2d_8], dim=1)
        conv2d_6 = conv2d_7 = conv2d_8 = None
        reshape_6 = conv_v_img_2.reshape(1, 8, 16, 3136)
        conv_v_img_2 = None
        conv_v_img_3 = reshape_6.transpose(-1, -2)
        reshape_6 = None
        EV_hat_2 = q_img_1 * conv_v_img_3
        q_img_1 = conv_v_img_3 = None
        EV_hat_3 = torch._C._nn.pad(EV_hat_2, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_2 = None
        mul_3 = 0.25 * factor_att_3
        factor_att_3 = None
        x_24 = mul_3 + EV_hat_3
        mul_3 = EV_hat_3 = None
        transpose_12 = x_24.transpose(1, 2)
        x_24 = None
        x_25 = transpose_12.reshape(1, 3137, 128)
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
            (128,),
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
        feat_2 = transpose_13.view(1, 128, 56, 56)
        transpose_13 = None
        conv2d_9 = torch.conv2d(
            feat_2,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
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
            (128,),
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
        reshape_8 = linear_8.reshape(1, 3137, 3, 8, 16)
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
        v_img_5 = transpose_16.reshape(1, 128, 56, 56)
        transpose_16 = None
        split_2 = torch.functional.split(v_img_5, [32, 48, 48], dim=1)
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
            32,
        )
        getitem_35 = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_11 = torch.conv2d(
            getitem_36,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            48,
        )
        getitem_36 = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_12 = torch.conv2d(
            getitem_37,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            48,
        )
        getitem_37 = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
        conv_v_img_4 = torch.cat([conv2d_10, conv2d_11, conv2d_12], dim=1)
        conv2d_10 = conv2d_11 = conv2d_12 = None
        reshape_10 = conv_v_img_4.reshape(1, 8, 16, 3136)
        conv_v_img_4 = None
        conv_v_img_5 = reshape_10.transpose(-1, -2)
        reshape_10 = None
        EV_hat_4 = q_img_2 * conv_v_img_5
        q_img_2 = conv_v_img_5 = None
        EV_hat_5 = torch._C._nn.pad(EV_hat_4, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_4 = None
        mul_5 = 0.25 * factor_att_5
        factor_att_5 = None
        x_40 = mul_5 + EV_hat_5
        mul_5 = EV_hat_5 = None
        transpose_18 = x_40.transpose(1, 2)
        x_40 = None
        x_41 = transpose_18.reshape(1, 3137, 128)
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
            (128,),
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
            (256,),
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
        feat_3 = transpose_20.view(1, 256, 28, 28)
        transpose_20 = None
        conv2d_14 = torch.conv2d(
            feat_3,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
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
            (256,),
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
        reshape_13 = linear_12.reshape(1, 785, 3, 8, 32)
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
        v_img_7 = transpose_23.reshape(1, 256, 28, 28)
        transpose_23 = None
        split_3 = torch.functional.split(v_img_7, [64, 96, 96], dim=1)
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
            64,
        )
        getitem_46 = None
        conv2d_16 = torch.conv2d(
            getitem_47,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        getitem_47 = None
        conv2d_17 = torch.conv2d(
            getitem_48,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            96,
        )
        getitem_48 = None
        conv_v_img_6 = torch.cat([conv2d_15, conv2d_16, conv2d_17], dim=1)
        conv2d_15 = conv2d_16 = conv2d_17 = None
        reshape_15 = conv_v_img_6.reshape(1, 8, 32, 784)
        conv_v_img_6 = None
        conv_v_img_7 = reshape_15.transpose(-1, -2)
        reshape_15 = None
        EV_hat_6 = q_img_3 * conv_v_img_7
        q_img_3 = conv_v_img_7 = None
        EV_hat_7 = torch._C._nn.pad(EV_hat_6, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_6 = None
        mul_7 = 0.1767766952966369 * factor_att_7
        factor_att_7 = None
        x_60 = mul_7 + EV_hat_7
        mul_7 = EV_hat_7 = None
        transpose_25 = x_60.transpose(1, 2)
        x_60 = None
        x_61 = transpose_25.reshape(1, 785, 256)
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
            (256,),
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
        feat_4 = transpose_26.view(1, 256, 28, 28)
        transpose_26 = None
        conv2d_18 = torch.conv2d(
            feat_4,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
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
            (256,),
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
        reshape_17 = linear_16.reshape(1, 785, 3, 8, 32)
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
        v_img_9 = transpose_29.reshape(1, 256, 28, 28)
        transpose_29 = None
        split_4 = torch.functional.split(v_img_9, [64, 96, 96], dim=1)
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
            64,
        )
        getitem_56 = None
        conv2d_20 = torch.conv2d(
            getitem_57,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        getitem_57 = None
        conv2d_21 = torch.conv2d(
            getitem_58,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            96,
        )
        getitem_58 = None
        conv_v_img_8 = torch.cat([conv2d_19, conv2d_20, conv2d_21], dim=1)
        conv2d_19 = conv2d_20 = conv2d_21 = None
        reshape_19 = conv_v_img_8.reshape(1, 8, 32, 784)
        conv_v_img_8 = None
        conv_v_img_9 = reshape_19.transpose(-1, -2)
        reshape_19 = None
        EV_hat_8 = q_img_4 * conv_v_img_9
        q_img_4 = conv_v_img_9 = None
        EV_hat_9 = torch._C._nn.pad(EV_hat_8, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_8 = None
        mul_9 = 0.1767766952966369 * factor_att_9
        factor_att_9 = None
        x_76 = mul_9 + EV_hat_9
        mul_9 = EV_hat_9 = None
        transpose_31 = x_76.transpose(1, 2)
        x_76 = None
        x_77 = transpose_31.reshape(1, 785, 256)
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
            (256,),
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
        feat_5 = transpose_32.view(1, 256, 28, 28)
        transpose_32 = None
        conv2d_22 = torch.conv2d(
            feat_5,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
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
            (256,),
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
        reshape_21 = linear_20.reshape(1, 785, 3, 8, 32)
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
        v_img_11 = transpose_35.reshape(1, 256, 28, 28)
        transpose_35 = None
        split_5 = torch.functional.split(v_img_11, [64, 96, 96], dim=1)
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
            64,
        )
        getitem_66 = None
        conv2d_24 = torch.conv2d(
            getitem_67,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        getitem_67 = None
        conv2d_25 = torch.conv2d(
            getitem_68,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            96,
        )
        getitem_68 = None
        conv_v_img_10 = torch.cat([conv2d_23, conv2d_24, conv2d_25], dim=1)
        conv2d_23 = conv2d_24 = conv2d_25 = None
        reshape_23 = conv_v_img_10.reshape(1, 8, 32, 784)
        conv_v_img_10 = None
        conv_v_img_11 = reshape_23.transpose(-1, -2)
        reshape_23 = None
        EV_hat_10 = q_img_5 * conv_v_img_11
        q_img_5 = conv_v_img_11 = None
        EV_hat_11 = torch._C._nn.pad(EV_hat_10, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_10 = None
        mul_11 = 0.1767766952966369 * factor_att_11
        factor_att_11 = None
        x_92 = mul_11 + EV_hat_11
        mul_11 = EV_hat_11 = None
        transpose_37 = x_92.transpose(1, 2)
        x_92 = None
        x_93 = transpose_37.reshape(1, 785, 256)
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
            (256,),
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
        feat_6 = transpose_38.view(1, 256, 28, 28)
        transpose_38 = None
        conv2d_26 = torch.conv2d(
            feat_6,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
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
            (256,),
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
        reshape_25 = linear_24.reshape(1, 785, 3, 8, 32)
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
        v_img_13 = transpose_41.reshape(1, 256, 28, 28)
        transpose_41 = None
        split_6 = torch.functional.split(v_img_13, [64, 96, 96], dim=1)
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
            64,
        )
        getitem_76 = None
        conv2d_28 = torch.conv2d(
            getitem_77,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        getitem_77 = None
        conv2d_29 = torch.conv2d(
            getitem_78,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            96,
        )
        getitem_78 = None
        conv_v_img_12 = torch.cat([conv2d_27, conv2d_28, conv2d_29], dim=1)
        conv2d_27 = conv2d_28 = conv2d_29 = None
        reshape_27 = conv_v_img_12.reshape(1, 8, 32, 784)
        conv_v_img_12 = None
        conv_v_img_13 = reshape_27.transpose(-1, -2)
        reshape_27 = None
        EV_hat_12 = q_img_6 * conv_v_img_13
        q_img_6 = conv_v_img_13 = None
        EV_hat_13 = torch._C._nn.pad(EV_hat_12, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_12 = None
        mul_13 = 0.1767766952966369 * factor_att_13
        factor_att_13 = None
        x_108 = mul_13 + EV_hat_13
        mul_13 = EV_hat_13 = None
        transpose_43 = x_108.transpose(1, 2)
        x_108 = None
        x_109 = transpose_43.reshape(1, 785, 256)
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
            (256,),
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
        cls_token_7 = x_119[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_7 = x_119[(slice(None, None, None), slice(1, None, None))]
        x_119 = None
        transpose_44 = img_tokens_7.transpose(1, 2)
        img_tokens_7 = None
        feat_7 = transpose_44.view(1, 256, 28, 28)
        transpose_44 = None
        conv2d_30 = torch.conv2d(
            feat_7,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_120 = conv2d_30 + feat_7
        conv2d_30 = feat_7 = None
        flatten_9 = x_120.flatten(2)
        x_120 = None
        x_121 = flatten_9.transpose(1, 2)
        flatten_9 = None
        x_122 = torch.cat((cls_token_7, x_121), dim=1)
        cls_token_7 = x_121 = None
        x_123 = torch.nn.functional.layer_norm(
            x_122,
            (256,),
            l_self_modules_serial_blocks2_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_4_modules_norm1_parameters_bias_
        ) = None
        linear_28 = torch._C._nn.linear(
            x_123,
            l_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_123 = l_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_29 = linear_28.reshape(1, 785, 3, 8, 32)
        linear_28 = None
        qkv_7 = reshape_29.permute(2, 0, 3, 1, 4)
        reshape_29 = None
        unbind_7 = qkv_7.unbind(0)
        qkv_7 = None
        q_7 = unbind_7[0]
        k_7 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        k_softmax_7 = k_7.softmax(dim=2)
        k_7 = None
        transpose_46 = k_softmax_7.transpose(-1, -2)
        k_softmax_7 = None
        factor_att_14 = transpose_46 @ v_7
        transpose_46 = None
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
        transpose_47 = v_img_14.transpose(-1, -2)
        v_img_14 = None
        v_img_15 = transpose_47.reshape(1, 256, 28, 28)
        transpose_47 = None
        split_7 = torch.functional.split(v_img_15, [64, 96, 96], dim=1)
        v_img_15 = None
        getitem_86 = split_7[0]
        getitem_87 = split_7[1]
        getitem_88 = split_7[2]
        split_7 = None
        conv2d_31 = torch.conv2d(
            getitem_86,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        getitem_86 = None
        conv2d_32 = torch.conv2d(
            getitem_87,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        getitem_87 = None
        conv2d_33 = torch.conv2d(
            getitem_88,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            96,
        )
        getitem_88 = None
        conv_v_img_14 = torch.cat([conv2d_31, conv2d_32, conv2d_33], dim=1)
        conv2d_31 = conv2d_32 = conv2d_33 = None
        reshape_31 = conv_v_img_14.reshape(1, 8, 32, 784)
        conv_v_img_14 = None
        conv_v_img_15 = reshape_31.transpose(-1, -2)
        reshape_31 = None
        EV_hat_14 = q_img_7 * conv_v_img_15
        q_img_7 = conv_v_img_15 = None
        EV_hat_15 = torch._C._nn.pad(EV_hat_14, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_14 = None
        mul_15 = 0.1767766952966369 * factor_att_15
        factor_att_15 = None
        x_124 = mul_15 + EV_hat_15
        mul_15 = EV_hat_15 = None
        transpose_49 = x_124.transpose(1, 2)
        x_124 = None
        x_125 = transpose_49.reshape(1, 785, 256)
        transpose_49 = None
        x_126 = torch._C._nn.linear(
            x_125,
            l_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_125 = l_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks2_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_127 = torch.nn.functional.dropout(x_126, 0.0, False, False)
        x_126 = None
        x_128 = x_122 + x_127
        x_122 = x_127 = None
        x_129 = torch.nn.functional.layer_norm(
            x_128,
            (256,),
            l_self_modules_serial_blocks2_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_130 = torch._C._nn.linear(
            x_129,
            l_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_129 = l_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_131 = torch._C._nn.gelu(x_130, approximate="none")
        x_130 = None
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        x_133 = torch._C._nn.linear(
            x_132,
            l_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_132 = l_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks2_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_134 = torch.nn.functional.dropout(x_133, 0.0, False, False)
        x_133 = None
        x_135 = x_128 + x_134
        x_128 = x_134 = None
        cls_token_8 = x_135[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_8 = x_135[(slice(None, None, None), slice(1, None, None))]
        x_135 = None
        transpose_50 = img_tokens_8.transpose(1, 2)
        img_tokens_8 = None
        feat_8 = transpose_50.view(1, 256, 28, 28)
        transpose_50 = None
        conv2d_34 = torch.conv2d(
            feat_8,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_ = (None)
        x_136 = conv2d_34 + feat_8
        conv2d_34 = feat_8 = None
        flatten_10 = x_136.flatten(2)
        x_136 = None
        x_137 = flatten_10.transpose(1, 2)
        flatten_10 = None
        x_138 = torch.cat((cls_token_8, x_137), dim=1)
        cls_token_8 = x_137 = None
        x_139 = torch.nn.functional.layer_norm(
            x_138,
            (256,),
            l_self_modules_serial_blocks2_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_5_modules_norm1_parameters_bias_
        ) = None
        linear_32 = torch._C._nn.linear(
            x_139,
            l_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_139 = l_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_33 = linear_32.reshape(1, 785, 3, 8, 32)
        linear_32 = None
        qkv_8 = reshape_33.permute(2, 0, 3, 1, 4)
        reshape_33 = None
        unbind_8 = qkv_8.unbind(0)
        qkv_8 = None
        q_8 = unbind_8[0]
        k_8 = unbind_8[1]
        v_8 = unbind_8[2]
        unbind_8 = None
        k_softmax_8 = k_8.softmax(dim=2)
        k_8 = None
        transpose_52 = k_softmax_8.transpose(-1, -2)
        k_softmax_8 = None
        factor_att_16 = transpose_52 @ v_8
        transpose_52 = None
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
        transpose_53 = v_img_16.transpose(-1, -2)
        v_img_16 = None
        v_img_17 = transpose_53.reshape(1, 256, 28, 28)
        transpose_53 = None
        split_8 = torch.functional.split(v_img_17, [64, 96, 96], dim=1)
        v_img_17 = None
        getitem_96 = split_8[0]
        getitem_97 = split_8[1]
        getitem_98 = split_8[2]
        split_8 = None
        conv2d_35 = torch.conv2d(
            getitem_96,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        getitem_96 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_36 = torch.conv2d(
            getitem_97,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        getitem_97 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_37 = torch.conv2d(
            getitem_98,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            96,
        )
        getitem_98 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
        conv_v_img_16 = torch.cat([conv2d_35, conv2d_36, conv2d_37], dim=1)
        conv2d_35 = conv2d_36 = conv2d_37 = None
        reshape_35 = conv_v_img_16.reshape(1, 8, 32, 784)
        conv_v_img_16 = None
        conv_v_img_17 = reshape_35.transpose(-1, -2)
        reshape_35 = None
        EV_hat_16 = q_img_8 * conv_v_img_17
        q_img_8 = conv_v_img_17 = None
        EV_hat_17 = torch._C._nn.pad(EV_hat_16, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_16 = None
        mul_17 = 0.1767766952966369 * factor_att_17
        factor_att_17 = None
        x_140 = mul_17 + EV_hat_17
        mul_17 = EV_hat_17 = None
        transpose_55 = x_140.transpose(1, 2)
        x_140 = None
        x_141 = transpose_55.reshape(1, 785, 256)
        transpose_55 = None
        x_142 = torch._C._nn.linear(
            x_141,
            l_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_141 = l_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks2_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_143 = torch.nn.functional.dropout(x_142, 0.0, False, False)
        x_142 = None
        x_144 = x_138 + x_143
        x_138 = x_143 = None
        x_145 = torch.nn.functional.layer_norm(
            x_144,
            (256,),
            l_self_modules_serial_blocks2_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks2_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks2_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_146 = torch._C._nn.linear(
            x_145,
            l_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_145 = l_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_147 = torch._C._nn.gelu(x_146, approximate="none")
        x_146 = None
        x_148 = torch.nn.functional.dropout(x_147, 0.0, False, False)
        x_147 = None
        x_149 = torch._C._nn.linear(
            x_148,
            l_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_148 = l_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks2_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_150 = torch.nn.functional.dropout(x_149, 0.0, False, False)
        x_149 = None
        x_151 = x_144 + x_150
        x_144 = x_150 = None
        getitem_99 = x_151[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        x_151 = None
        reshape_37 = getitem_99.reshape(1, 28, 28, -1)
        getitem_99 = None
        permute_10 = reshape_37.permute(0, 3, 1, 2)
        reshape_37 = None
        x2_nocls = permute_10.contiguous()
        permute_10 = None
        x_152 = torch.conv2d(
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
        flatten_11 = x_152.flatten(2)
        x_152 = None
        x_153 = flatten_11.transpose(1, 2)
        flatten_11 = None
        x_154 = torch.nn.functional.layer_norm(
            x_153,
            (320,),
            l_self_modules_patch_embed3_modules_norm_parameters_weight_,
            l_self_modules_patch_embed3_modules_norm_parameters_bias_,
            1e-05,
        )
        x_153 = (
            l_self_modules_patch_embed3_modules_norm_parameters_weight_
        ) = l_self_modules_patch_embed3_modules_norm_parameters_bias_ = None
        cls_tokens_2 = l_self_parameters_cls_token3_.expand(1, -1, -1)
        l_self_parameters_cls_token3_ = None
        x_155 = torch.cat((cls_tokens_2, x_154), dim=1)
        cls_tokens_2 = x_154 = None
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
            l_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear_36 = torch._C._nn.linear(
            x_159,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_159 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
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
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_161 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_163 = torch.nn.functional.dropout(x_162, 0.0, False, False)
        x_162 = None
        x_164 = x_158 + x_163
        x_158 = x_163 = None
        x_165 = torch.nn.functional.layer_norm(
            x_164,
            (320,),
            l_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_166 = torch._C._nn.linear(
            x_165,
            l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_165 = l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_167 = torch._C._nn.gelu(x_166, approximate="none")
        x_166 = None
        x_168 = torch.nn.functional.dropout(x_167, 0.0, False, False)
        x_167 = None
        x_169 = torch._C._nn.linear(
            x_168,
            l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_168 = l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
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
            l_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_1_modules_norm1_parameters_bias_
        ) = None
        linear_40 = torch._C._nn.linear(
            x_175,
            l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_175 = l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
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
            l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_177 = l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_179 = torch.nn.functional.dropout(x_178, 0.0, False, False)
        x_178 = None
        x_180 = x_174 + x_179
        x_174 = x_179 = None
        x_181 = torch.nn.functional.layer_norm(
            x_180,
            (320,),
            l_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_182 = torch._C._nn.linear(
            x_181,
            l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_181 = l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_183 = torch._C._nn.gelu(x_182, approximate="none")
        x_182 = None
        x_184 = torch.nn.functional.dropout(x_183, 0.0, False, False)
        x_183 = None
        x_185 = torch._C._nn.linear(
            x_184,
            l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_184 = l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
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
            l_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_2_modules_norm1_parameters_bias_
        ) = None
        linear_44 = torch._C._nn.linear(
            x_191,
            l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_191 = l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
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
            l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_193 = l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_195 = torch.nn.functional.dropout(x_194, 0.0, False, False)
        x_194 = None
        x_196 = x_190 + x_195
        x_190 = x_195 = None
        x_197 = torch.nn.functional.layer_norm(
            x_196,
            (320,),
            l_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_198 = torch._C._nn.linear(
            x_197,
            l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_197 = l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_199 = torch._C._nn.gelu(x_198, approximate="none")
        x_198 = None
        x_200 = torch.nn.functional.dropout(x_199, 0.0, False, False)
        x_199 = None
        x_201 = torch._C._nn.linear(
            x_200,
            l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_200 = l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
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
            l_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_3_modules_norm1_parameters_bias_
        ) = None
        linear_48 = torch._C._nn.linear(
            x_207,
            l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_207 = l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
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
        getitem_137 = None
        conv2d_53 = torch.conv2d(
            getitem_138,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_138 = None
        conv2d_54 = torch.conv2d(
            getitem_139,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_139 = None
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
            l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_209 = l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_211 = torch.nn.functional.dropout(x_210, 0.0, False, False)
        x_210 = None
        x_212 = x_206 + x_211
        x_206 = x_211 = None
        x_213 = torch.nn.functional.layer_norm(
            x_212,
            (320,),
            l_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_214 = torch._C._nn.linear(
            x_213,
            l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_213 = l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_215 = torch._C._nn.gelu(x_214, approximate="none")
        x_214 = None
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        x_217 = torch._C._nn.linear(
            x_216,
            l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_216 = l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_218 = torch.nn.functional.dropout(x_217, 0.0, False, False)
        x_217 = None
        x_219 = x_212 + x_218
        x_212 = x_218 = None
        cls_token_13 = x_219[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_13 = x_219[(slice(None, None, None), slice(1, None, None))]
        x_219 = None
        transpose_81 = img_tokens_13.transpose(1, 2)
        img_tokens_13 = None
        feat_13 = transpose_81.view(1, 320, 14, 14)
        transpose_81 = None
        conv2d_55 = torch.conv2d(
            feat_13,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        x_220 = conv2d_55 + feat_13
        conv2d_55 = feat_13 = None
        flatten_16 = x_220.flatten(2)
        x_220 = None
        x_221 = flatten_16.transpose(1, 2)
        flatten_16 = None
        x_222 = torch.cat((cls_token_13, x_221), dim=1)
        cls_token_13 = x_221 = None
        x_223 = torch.nn.functional.layer_norm(
            x_222,
            (320,),
            l_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_4_modules_norm1_parameters_bias_
        ) = None
        linear_52 = torch._C._nn.linear(
            x_223,
            l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_223 = l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_54 = linear_52.reshape(1, 197, 3, 8, 40)
        linear_52 = None
        qkv_13 = reshape_54.permute(2, 0, 3, 1, 4)
        reshape_54 = None
        unbind_13 = qkv_13.unbind(0)
        qkv_13 = None
        q_13 = unbind_13[0]
        k_13 = unbind_13[1]
        v_13 = unbind_13[2]
        unbind_13 = None
        k_softmax_13 = k_13.softmax(dim=2)
        k_13 = None
        transpose_83 = k_softmax_13.transpose(-1, -2)
        k_softmax_13 = None
        factor_att_26 = transpose_83 @ v_13
        transpose_83 = None
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
        transpose_84 = v_img_26.transpose(-1, -2)
        v_img_26 = None
        v_img_27 = transpose_84.reshape(1, 320, 14, 14)
        transpose_84 = None
        split_13 = torch.functional.split(v_img_27, [80, 120, 120], dim=1)
        v_img_27 = None
        getitem_147 = split_13[0]
        getitem_148 = split_13[1]
        getitem_149 = split_13[2]
        split_13 = None
        conv2d_56 = torch.conv2d(
            getitem_147,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        getitem_147 = None
        conv2d_57 = torch.conv2d(
            getitem_148,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_148 = None
        conv2d_58 = torch.conv2d(
            getitem_149,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_149 = None
        conv_v_img_26 = torch.cat([conv2d_56, conv2d_57, conv2d_58], dim=1)
        conv2d_56 = conv2d_57 = conv2d_58 = None
        reshape_56 = conv_v_img_26.reshape(1, 8, 40, 196)
        conv_v_img_26 = None
        conv_v_img_27 = reshape_56.transpose(-1, -2)
        reshape_56 = None
        EV_hat_26 = q_img_13 * conv_v_img_27
        q_img_13 = conv_v_img_27 = None
        EV_hat_27 = torch._C._nn.pad(EV_hat_26, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_26 = None
        mul_27 = 0.15811388300841897 * factor_att_27
        factor_att_27 = None
        x_224 = mul_27 + EV_hat_27
        mul_27 = EV_hat_27 = None
        transpose_86 = x_224.transpose(1, 2)
        x_224 = None
        x_225 = transpose_86.reshape(1, 197, 320)
        transpose_86 = None
        x_226 = torch._C._nn.linear(
            x_225,
            l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_225 = l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_227 = torch.nn.functional.dropout(x_226, 0.0, False, False)
        x_226 = None
        x_228 = x_222 + x_227
        x_222 = x_227 = None
        x_229 = torch.nn.functional.layer_norm(
            x_228,
            (320,),
            l_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_230 = torch._C._nn.linear(
            x_229,
            l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_229 = l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_231 = torch._C._nn.gelu(x_230, approximate="none")
        x_230 = None
        x_232 = torch.nn.functional.dropout(x_231, 0.0, False, False)
        x_231 = None
        x_233 = torch._C._nn.linear(
            x_232,
            l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_232 = l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_234 = torch.nn.functional.dropout(x_233, 0.0, False, False)
        x_233 = None
        x_235 = x_228 + x_234
        x_228 = x_234 = None
        cls_token_14 = x_235[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_14 = x_235[(slice(None, None, None), slice(1, None, None))]
        x_235 = None
        transpose_87 = img_tokens_14.transpose(1, 2)
        img_tokens_14 = None
        feat_14 = transpose_87.view(1, 320, 14, 14)
        transpose_87 = None
        conv2d_59 = torch.conv2d(
            feat_14,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        x_236 = conv2d_59 + feat_14
        conv2d_59 = feat_14 = None
        flatten_17 = x_236.flatten(2)
        x_236 = None
        x_237 = flatten_17.transpose(1, 2)
        flatten_17 = None
        x_238 = torch.cat((cls_token_14, x_237), dim=1)
        cls_token_14 = x_237 = None
        x_239 = torch.nn.functional.layer_norm(
            x_238,
            (320,),
            l_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_5_modules_norm1_parameters_bias_
        ) = None
        linear_56 = torch._C._nn.linear(
            x_239,
            l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_239 = l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_58 = linear_56.reshape(1, 197, 3, 8, 40)
        linear_56 = None
        qkv_14 = reshape_58.permute(2, 0, 3, 1, 4)
        reshape_58 = None
        unbind_14 = qkv_14.unbind(0)
        qkv_14 = None
        q_14 = unbind_14[0]
        k_14 = unbind_14[1]
        v_14 = unbind_14[2]
        unbind_14 = None
        k_softmax_14 = k_14.softmax(dim=2)
        k_14 = None
        transpose_89 = k_softmax_14.transpose(-1, -2)
        k_softmax_14 = None
        factor_att_28 = transpose_89 @ v_14
        transpose_89 = None
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
        transpose_90 = v_img_28.transpose(-1, -2)
        v_img_28 = None
        v_img_29 = transpose_90.reshape(1, 320, 14, 14)
        transpose_90 = None
        split_14 = torch.functional.split(v_img_29, [80, 120, 120], dim=1)
        v_img_29 = None
        getitem_157 = split_14[0]
        getitem_158 = split_14[1]
        getitem_159 = split_14[2]
        split_14 = None
        conv2d_60 = torch.conv2d(
            getitem_157,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        getitem_157 = None
        conv2d_61 = torch.conv2d(
            getitem_158,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_158 = None
        conv2d_62 = torch.conv2d(
            getitem_159,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_159 = None
        conv_v_img_28 = torch.cat([conv2d_60, conv2d_61, conv2d_62], dim=1)
        conv2d_60 = conv2d_61 = conv2d_62 = None
        reshape_60 = conv_v_img_28.reshape(1, 8, 40, 196)
        conv_v_img_28 = None
        conv_v_img_29 = reshape_60.transpose(-1, -2)
        reshape_60 = None
        EV_hat_28 = q_img_14 * conv_v_img_29
        q_img_14 = conv_v_img_29 = None
        EV_hat_29 = torch._C._nn.pad(EV_hat_28, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_28 = None
        mul_29 = 0.15811388300841897 * factor_att_29
        factor_att_29 = None
        x_240 = mul_29 + EV_hat_29
        mul_29 = EV_hat_29 = None
        transpose_92 = x_240.transpose(1, 2)
        x_240 = None
        x_241 = transpose_92.reshape(1, 197, 320)
        transpose_92 = None
        x_242 = torch._C._nn.linear(
            x_241,
            l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_241 = l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_243 = torch.nn.functional.dropout(x_242, 0.0, False, False)
        x_242 = None
        x_244 = x_238 + x_243
        x_238 = x_243 = None
        x_245 = torch.nn.functional.layer_norm(
            x_244,
            (320,),
            l_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_246 = torch._C._nn.linear(
            x_245,
            l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_245 = l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_247 = torch._C._nn.gelu(x_246, approximate="none")
        x_246 = None
        x_248 = torch.nn.functional.dropout(x_247, 0.0, False, False)
        x_247 = None
        x_249 = torch._C._nn.linear(
            x_248,
            l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_248 = l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_250 = torch.nn.functional.dropout(x_249, 0.0, False, False)
        x_249 = None
        x_251 = x_244 + x_250
        x_244 = x_250 = None
        cls_token_15 = x_251[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_15 = x_251[(slice(None, None, None), slice(1, None, None))]
        x_251 = None
        transpose_93 = img_tokens_15.transpose(1, 2)
        img_tokens_15 = None
        feat_15 = transpose_93.view(1, 320, 14, 14)
        transpose_93 = None
        conv2d_63 = torch.conv2d(
            feat_15,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        x_252 = conv2d_63 + feat_15
        conv2d_63 = feat_15 = None
        flatten_18 = x_252.flatten(2)
        x_252 = None
        x_253 = flatten_18.transpose(1, 2)
        flatten_18 = None
        x_254 = torch.cat((cls_token_15, x_253), dim=1)
        cls_token_15 = x_253 = None
        x_255 = torch.nn.functional.layer_norm(
            x_254,
            (320,),
            l_self_modules_serial_blocks3_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_6_modules_norm1_parameters_bias_
        ) = None
        linear_60 = torch._C._nn.linear(
            x_255,
            l_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_255 = l_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_62 = linear_60.reshape(1, 197, 3, 8, 40)
        linear_60 = None
        qkv_15 = reshape_62.permute(2, 0, 3, 1, 4)
        reshape_62 = None
        unbind_15 = qkv_15.unbind(0)
        qkv_15 = None
        q_15 = unbind_15[0]
        k_15 = unbind_15[1]
        v_15 = unbind_15[2]
        unbind_15 = None
        k_softmax_15 = k_15.softmax(dim=2)
        k_15 = None
        transpose_95 = k_softmax_15.transpose(-1, -2)
        k_softmax_15 = None
        factor_att_30 = transpose_95 @ v_15
        transpose_95 = None
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
        transpose_96 = v_img_30.transpose(-1, -2)
        v_img_30 = None
        v_img_31 = transpose_96.reshape(1, 320, 14, 14)
        transpose_96 = None
        split_15 = torch.functional.split(v_img_31, [80, 120, 120], dim=1)
        v_img_31 = None
        getitem_167 = split_15[0]
        getitem_168 = split_15[1]
        getitem_169 = split_15[2]
        split_15 = None
        conv2d_64 = torch.conv2d(
            getitem_167,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        getitem_167 = None
        conv2d_65 = torch.conv2d(
            getitem_168,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_168 = None
        conv2d_66 = torch.conv2d(
            getitem_169,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_169 = None
        conv_v_img_30 = torch.cat([conv2d_64, conv2d_65, conv2d_66], dim=1)
        conv2d_64 = conv2d_65 = conv2d_66 = None
        reshape_64 = conv_v_img_30.reshape(1, 8, 40, 196)
        conv_v_img_30 = None
        conv_v_img_31 = reshape_64.transpose(-1, -2)
        reshape_64 = None
        EV_hat_30 = q_img_15 * conv_v_img_31
        q_img_15 = conv_v_img_31 = None
        EV_hat_31 = torch._C._nn.pad(EV_hat_30, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_30 = None
        mul_31 = 0.15811388300841897 * factor_att_31
        factor_att_31 = None
        x_256 = mul_31 + EV_hat_31
        mul_31 = EV_hat_31 = None
        transpose_98 = x_256.transpose(1, 2)
        x_256 = None
        x_257 = transpose_98.reshape(1, 197, 320)
        transpose_98 = None
        x_258 = torch._C._nn.linear(
            x_257,
            l_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_257 = l_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_6_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_259 = torch.nn.functional.dropout(x_258, 0.0, False, False)
        x_258 = None
        x_260 = x_254 + x_259
        x_254 = x_259 = None
        x_261 = torch.nn.functional.layer_norm(
            x_260,
            (320,),
            l_self_modules_serial_blocks3_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_6_modules_norm2_parameters_bias_
        ) = None
        x_262 = torch._C._nn.linear(
            x_261,
            l_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_261 = l_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_263 = torch._C._nn.gelu(x_262, approximate="none")
        x_262 = None
        x_264 = torch.nn.functional.dropout(x_263, 0.0, False, False)
        x_263 = None
        x_265 = torch._C._nn.linear(
            x_264,
            l_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_264 = l_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_266 = torch.nn.functional.dropout(x_265, 0.0, False, False)
        x_265 = None
        x_267 = x_260 + x_266
        x_260 = x_266 = None
        cls_token_16 = x_267[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_16 = x_267[(slice(None, None, None), slice(1, None, None))]
        x_267 = None
        transpose_99 = img_tokens_16.transpose(1, 2)
        img_tokens_16 = None
        feat_16 = transpose_99.view(1, 320, 14, 14)
        transpose_99 = None
        conv2d_67 = torch.conv2d(
            feat_16,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        x_268 = conv2d_67 + feat_16
        conv2d_67 = feat_16 = None
        flatten_19 = x_268.flatten(2)
        x_268 = None
        x_269 = flatten_19.transpose(1, 2)
        flatten_19 = None
        x_270 = torch.cat((cls_token_16, x_269), dim=1)
        cls_token_16 = x_269 = None
        x_271 = torch.nn.functional.layer_norm(
            x_270,
            (320,),
            l_self_modules_serial_blocks3_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_7_modules_norm1_parameters_bias_
        ) = None
        linear_64 = torch._C._nn.linear(
            x_271,
            l_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_271 = l_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_66 = linear_64.reshape(1, 197, 3, 8, 40)
        linear_64 = None
        qkv_16 = reshape_66.permute(2, 0, 3, 1, 4)
        reshape_66 = None
        unbind_16 = qkv_16.unbind(0)
        qkv_16 = None
        q_16 = unbind_16[0]
        k_16 = unbind_16[1]
        v_16 = unbind_16[2]
        unbind_16 = None
        k_softmax_16 = k_16.softmax(dim=2)
        k_16 = None
        transpose_101 = k_softmax_16.transpose(-1, -2)
        k_softmax_16 = None
        factor_att_32 = transpose_101 @ v_16
        transpose_101 = None
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
        transpose_102 = v_img_32.transpose(-1, -2)
        v_img_32 = None
        v_img_33 = transpose_102.reshape(1, 320, 14, 14)
        transpose_102 = None
        split_16 = torch.functional.split(v_img_33, [80, 120, 120], dim=1)
        v_img_33 = None
        getitem_177 = split_16[0]
        getitem_178 = split_16[1]
        getitem_179 = split_16[2]
        split_16 = None
        conv2d_68 = torch.conv2d(
            getitem_177,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        getitem_177 = None
        conv2d_69 = torch.conv2d(
            getitem_178,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_178 = None
        conv2d_70 = torch.conv2d(
            getitem_179,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_179 = None
        conv_v_img_32 = torch.cat([conv2d_68, conv2d_69, conv2d_70], dim=1)
        conv2d_68 = conv2d_69 = conv2d_70 = None
        reshape_68 = conv_v_img_32.reshape(1, 8, 40, 196)
        conv_v_img_32 = None
        conv_v_img_33 = reshape_68.transpose(-1, -2)
        reshape_68 = None
        EV_hat_32 = q_img_16 * conv_v_img_33
        q_img_16 = conv_v_img_33 = None
        EV_hat_33 = torch._C._nn.pad(EV_hat_32, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_32 = None
        mul_33 = 0.15811388300841897 * factor_att_33
        factor_att_33 = None
        x_272 = mul_33 + EV_hat_33
        mul_33 = EV_hat_33 = None
        transpose_104 = x_272.transpose(1, 2)
        x_272 = None
        x_273 = transpose_104.reshape(1, 197, 320)
        transpose_104 = None
        x_274 = torch._C._nn.linear(
            x_273,
            l_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_273 = l_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_7_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_275 = torch.nn.functional.dropout(x_274, 0.0, False, False)
        x_274 = None
        x_276 = x_270 + x_275
        x_270 = x_275 = None
        x_277 = torch.nn.functional.layer_norm(
            x_276,
            (320,),
            l_self_modules_serial_blocks3_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_7_modules_norm2_parameters_bias_
        ) = None
        x_278 = torch._C._nn.linear(
            x_277,
            l_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_277 = l_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_279 = torch._C._nn.gelu(x_278, approximate="none")
        x_278 = None
        x_280 = torch.nn.functional.dropout(x_279, 0.0, False, False)
        x_279 = None
        x_281 = torch._C._nn.linear(
            x_280,
            l_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_280 = l_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_282 = torch.nn.functional.dropout(x_281, 0.0, False, False)
        x_281 = None
        x_283 = x_276 + x_282
        x_276 = x_282 = None
        cls_token_17 = x_283[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_17 = x_283[(slice(None, None, None), slice(1, None, None))]
        x_283 = None
        transpose_105 = img_tokens_17.transpose(1, 2)
        img_tokens_17 = None
        feat_17 = transpose_105.view(1, 320, 14, 14)
        transpose_105 = None
        conv2d_71 = torch.conv2d(
            feat_17,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        x_284 = conv2d_71 + feat_17
        conv2d_71 = feat_17 = None
        flatten_20 = x_284.flatten(2)
        x_284 = None
        x_285 = flatten_20.transpose(1, 2)
        flatten_20 = None
        x_286 = torch.cat((cls_token_17, x_285), dim=1)
        cls_token_17 = x_285 = None
        x_287 = torch.nn.functional.layer_norm(
            x_286,
            (320,),
            l_self_modules_serial_blocks3_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_8_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_8_modules_norm1_parameters_bias_
        ) = None
        linear_68 = torch._C._nn.linear(
            x_287,
            l_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_287 = l_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_70 = linear_68.reshape(1, 197, 3, 8, 40)
        linear_68 = None
        qkv_17 = reshape_70.permute(2, 0, 3, 1, 4)
        reshape_70 = None
        unbind_17 = qkv_17.unbind(0)
        qkv_17 = None
        q_17 = unbind_17[0]
        k_17 = unbind_17[1]
        v_17 = unbind_17[2]
        unbind_17 = None
        k_softmax_17 = k_17.softmax(dim=2)
        k_17 = None
        transpose_107 = k_softmax_17.transpose(-1, -2)
        k_softmax_17 = None
        factor_att_34 = transpose_107 @ v_17
        transpose_107 = None
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
        transpose_108 = v_img_34.transpose(-1, -2)
        v_img_34 = None
        v_img_35 = transpose_108.reshape(1, 320, 14, 14)
        transpose_108 = None
        split_17 = torch.functional.split(v_img_35, [80, 120, 120], dim=1)
        v_img_35 = None
        getitem_187 = split_17[0]
        getitem_188 = split_17[1]
        getitem_189 = split_17[2]
        split_17 = None
        conv2d_72 = torch.conv2d(
            getitem_187,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        getitem_187 = None
        conv2d_73 = torch.conv2d(
            getitem_188,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_188 = None
        conv2d_74 = torch.conv2d(
            getitem_189,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_189 = None
        conv_v_img_34 = torch.cat([conv2d_72, conv2d_73, conv2d_74], dim=1)
        conv2d_72 = conv2d_73 = conv2d_74 = None
        reshape_72 = conv_v_img_34.reshape(1, 8, 40, 196)
        conv_v_img_34 = None
        conv_v_img_35 = reshape_72.transpose(-1, -2)
        reshape_72 = None
        EV_hat_34 = q_img_17 * conv_v_img_35
        q_img_17 = conv_v_img_35 = None
        EV_hat_35 = torch._C._nn.pad(EV_hat_34, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_34 = None
        mul_35 = 0.15811388300841897 * factor_att_35
        factor_att_35 = None
        x_288 = mul_35 + EV_hat_35
        mul_35 = EV_hat_35 = None
        transpose_110 = x_288.transpose(1, 2)
        x_288 = None
        x_289 = transpose_110.reshape(1, 197, 320)
        transpose_110 = None
        x_290 = torch._C._nn.linear(
            x_289,
            l_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_289 = l_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_8_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_291 = torch.nn.functional.dropout(x_290, 0.0, False, False)
        x_290 = None
        x_292 = x_286 + x_291
        x_286 = x_291 = None
        x_293 = torch.nn.functional.layer_norm(
            x_292,
            (320,),
            l_self_modules_serial_blocks3_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_8_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_8_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_8_modules_norm2_parameters_bias_
        ) = None
        x_294 = torch._C._nn.linear(
            x_293,
            l_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_293 = l_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_295 = torch._C._nn.gelu(x_294, approximate="none")
        x_294 = None
        x_296 = torch.nn.functional.dropout(x_295, 0.0, False, False)
        x_295 = None
        x_297 = torch._C._nn.linear(
            x_296,
            l_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_296 = l_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_298 = torch.nn.functional.dropout(x_297, 0.0, False, False)
        x_297 = None
        x_299 = x_292 + x_298
        x_292 = x_298 = None
        cls_token_18 = x_299[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_18 = x_299[(slice(None, None, None), slice(1, None, None))]
        x_299 = None
        transpose_111 = img_tokens_18.transpose(1, 2)
        img_tokens_18 = None
        feat_18 = transpose_111.view(1, 320, 14, 14)
        transpose_111 = None
        conv2d_75 = torch.conv2d(
            feat_18,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_ = (None)
        x_300 = conv2d_75 + feat_18
        conv2d_75 = feat_18 = None
        flatten_21 = x_300.flatten(2)
        x_300 = None
        x_301 = flatten_21.transpose(1, 2)
        flatten_21 = None
        x_302 = torch.cat((cls_token_18, x_301), dim=1)
        cls_token_18 = x_301 = None
        x_303 = torch.nn.functional.layer_norm(
            x_302,
            (320,),
            l_self_modules_serial_blocks3_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_9_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_9_modules_norm1_parameters_bias_
        ) = None
        linear_72 = torch._C._nn.linear(
            x_303,
            l_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_303 = l_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_74 = linear_72.reshape(1, 197, 3, 8, 40)
        linear_72 = None
        qkv_18 = reshape_74.permute(2, 0, 3, 1, 4)
        reshape_74 = None
        unbind_18 = qkv_18.unbind(0)
        qkv_18 = None
        q_18 = unbind_18[0]
        k_18 = unbind_18[1]
        v_18 = unbind_18[2]
        unbind_18 = None
        k_softmax_18 = k_18.softmax(dim=2)
        k_18 = None
        transpose_113 = k_softmax_18.transpose(-1, -2)
        k_softmax_18 = None
        factor_att_36 = transpose_113 @ v_18
        transpose_113 = None
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
        transpose_114 = v_img_36.transpose(-1, -2)
        v_img_36 = None
        v_img_37 = transpose_114.reshape(1, 320, 14, 14)
        transpose_114 = None
        split_18 = torch.functional.split(v_img_37, [80, 120, 120], dim=1)
        v_img_37 = None
        getitem_197 = split_18[0]
        getitem_198 = split_18[1]
        getitem_199 = split_18[2]
        split_18 = None
        conv2d_76 = torch.conv2d(
            getitem_197,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        getitem_197 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_77 = torch.conv2d(
            getitem_198,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_198 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_78 = torch.conv2d(
            getitem_199,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_199 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
        conv_v_img_36 = torch.cat([conv2d_76, conv2d_77, conv2d_78], dim=1)
        conv2d_76 = conv2d_77 = conv2d_78 = None
        reshape_76 = conv_v_img_36.reshape(1, 8, 40, 196)
        conv_v_img_36 = None
        conv_v_img_37 = reshape_76.transpose(-1, -2)
        reshape_76 = None
        EV_hat_36 = q_img_18 * conv_v_img_37
        q_img_18 = conv_v_img_37 = None
        EV_hat_37 = torch._C._nn.pad(EV_hat_36, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_36 = None
        mul_37 = 0.15811388300841897 * factor_att_37
        factor_att_37 = None
        x_304 = mul_37 + EV_hat_37
        mul_37 = EV_hat_37 = None
        transpose_116 = x_304.transpose(1, 2)
        x_304 = None
        x_305 = transpose_116.reshape(1, 197, 320)
        transpose_116 = None
        x_306 = torch._C._nn.linear(
            x_305,
            l_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_305 = l_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_9_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_307 = torch.nn.functional.dropout(x_306, 0.0, False, False)
        x_306 = None
        x_308 = x_302 + x_307
        x_302 = x_307 = None
        x_309 = torch.nn.functional.layer_norm(
            x_308,
            (320,),
            l_self_modules_serial_blocks3_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_9_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks3_modules_9_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks3_modules_9_modules_norm2_parameters_bias_
        ) = None
        x_310 = torch._C._nn.linear(
            x_309,
            l_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_309 = l_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_311 = torch._C._nn.gelu(x_310, approximate="none")
        x_310 = None
        x_312 = torch.nn.functional.dropout(x_311, 0.0, False, False)
        x_311 = None
        x_313 = torch._C._nn.linear(
            x_312,
            l_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_312 = l_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks3_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_314 = torch.nn.functional.dropout(x_313, 0.0, False, False)
        x_313 = None
        x_315 = x_308 + x_314
        x_308 = x_314 = None
        getitem_200 = x_315[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        x_315 = None
        reshape_78 = getitem_200.reshape(1, 14, 14, -1)
        getitem_200 = None
        permute_21 = reshape_78.permute(0, 3, 1, 2)
        reshape_78 = None
        x3_nocls = permute_21.contiguous()
        permute_21 = None
        x_316 = torch.conv2d(
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
        flatten_22 = x_316.flatten(2)
        x_316 = None
        x_317 = flatten_22.transpose(1, 2)
        flatten_22 = None
        x_318 = torch.nn.functional.layer_norm(
            x_317,
            (512,),
            l_self_modules_patch_embed4_modules_norm_parameters_weight_,
            l_self_modules_patch_embed4_modules_norm_parameters_bias_,
            1e-05,
        )
        x_317 = (
            l_self_modules_patch_embed4_modules_norm_parameters_weight_
        ) = l_self_modules_patch_embed4_modules_norm_parameters_bias_ = None
        cls_tokens_3 = l_self_parameters_cls_token4_.expand(1, -1, -1)
        l_self_parameters_cls_token4_ = None
        x_319 = torch.cat((cls_tokens_3, x_318), dim=1)
        cls_tokens_3 = x_318 = None
        cls_token_19 = x_319[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_19 = x_319[(slice(None, None, None), slice(1, None, None))]
        x_319 = None
        transpose_118 = img_tokens_19.transpose(1, 2)
        img_tokens_19 = None
        feat_19 = transpose_118.view(1, 512, 7, 7)
        transpose_118 = None
        conv2d_80 = torch.conv2d(
            feat_19,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_320 = conv2d_80 + feat_19
        conv2d_80 = feat_19 = None
        flatten_23 = x_320.flatten(2)
        x_320 = None
        x_321 = flatten_23.transpose(1, 2)
        flatten_23 = None
        x_322 = torch.cat((cls_token_19, x_321), dim=1)
        cls_token_19 = x_321 = None
        x_323 = torch.nn.functional.layer_norm(
            x_322,
            (512,),
            l_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear_76 = torch._C._nn.linear(
            x_323,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_323 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_79 = linear_76.reshape(1, 50, 3, 8, 64)
        linear_76 = None
        qkv_19 = reshape_79.permute(2, 0, 3, 1, 4)
        reshape_79 = None
        unbind_19 = qkv_19.unbind(0)
        qkv_19 = None
        q_19 = unbind_19[0]
        k_19 = unbind_19[1]
        v_19 = unbind_19[2]
        unbind_19 = None
        k_softmax_19 = k_19.softmax(dim=2)
        k_19 = None
        transpose_120 = k_softmax_19.transpose(-1, -2)
        k_softmax_19 = None
        factor_att_38 = transpose_120 @ v_19
        transpose_120 = None
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
        transpose_121 = v_img_38.transpose(-1, -2)
        v_img_38 = None
        v_img_39 = transpose_121.reshape(1, 512, 7, 7)
        transpose_121 = None
        split_19 = torch.functional.split(v_img_39, [128, 192, 192], dim=1)
        v_img_39 = None
        getitem_208 = split_19[0]
        getitem_209 = split_19[1]
        getitem_210 = split_19[2]
        split_19 = None
        conv2d_81 = torch.conv2d(
            getitem_208,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        getitem_208 = None
        conv2d_82 = torch.conv2d(
            getitem_209,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        getitem_209 = None
        conv2d_83 = torch.conv2d(
            getitem_210,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        getitem_210 = None
        conv_v_img_38 = torch.cat([conv2d_81, conv2d_82, conv2d_83], dim=1)
        conv2d_81 = conv2d_82 = conv2d_83 = None
        reshape_81 = conv_v_img_38.reshape(1, 8, 64, 49)
        conv_v_img_38 = None
        conv_v_img_39 = reshape_81.transpose(-1, -2)
        reshape_81 = None
        EV_hat_38 = q_img_19 * conv_v_img_39
        q_img_19 = conv_v_img_39 = None
        EV_hat_39 = torch._C._nn.pad(EV_hat_38, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_38 = None
        mul_39 = 0.125 * factor_att_39
        factor_att_39 = None
        x_324 = mul_39 + EV_hat_39
        mul_39 = EV_hat_39 = None
        transpose_123 = x_324.transpose(1, 2)
        x_324 = None
        x_325 = transpose_123.reshape(1, 50, 512)
        transpose_123 = None
        x_326 = torch._C._nn.linear(
            x_325,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_325 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_327 = torch.nn.functional.dropout(x_326, 0.0, False, False)
        x_326 = None
        x_328 = x_322 + x_327
        x_322 = x_327 = None
        x_329 = torch.nn.functional.layer_norm(
            x_328,
            (512,),
            l_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_330 = torch._C._nn.linear(
            x_329,
            l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_329 = l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_331 = torch._C._nn.gelu(x_330, approximate="none")
        x_330 = None
        x_332 = torch.nn.functional.dropout(x_331, 0.0, False, False)
        x_331 = None
        x_333 = torch._C._nn.linear(
            x_332,
            l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_332 = l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_334 = torch.nn.functional.dropout(x_333, 0.0, False, False)
        x_333 = None
        x_335 = x_328 + x_334
        x_328 = x_334 = None
        cls_token_20 = x_335[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_20 = x_335[(slice(None, None, None), slice(1, None, None))]
        x_335 = None
        transpose_124 = img_tokens_20.transpose(1, 2)
        img_tokens_20 = None
        feat_20 = transpose_124.view(1, 512, 7, 7)
        transpose_124 = None
        conv2d_84 = torch.conv2d(
            feat_20,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_336 = conv2d_84 + feat_20
        conv2d_84 = feat_20 = None
        flatten_24 = x_336.flatten(2)
        x_336 = None
        x_337 = flatten_24.transpose(1, 2)
        flatten_24 = None
        x_338 = torch.cat((cls_token_20, x_337), dim=1)
        cls_token_20 = x_337 = None
        x_339 = torch.nn.functional.layer_norm(
            x_338,
            (512,),
            l_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_1_modules_norm1_parameters_bias_
        ) = None
        linear_80 = torch._C._nn.linear(
            x_339,
            l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_339 = l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_83 = linear_80.reshape(1, 50, 3, 8, 64)
        linear_80 = None
        qkv_20 = reshape_83.permute(2, 0, 3, 1, 4)
        reshape_83 = None
        unbind_20 = qkv_20.unbind(0)
        qkv_20 = None
        q_20 = unbind_20[0]
        k_20 = unbind_20[1]
        v_20 = unbind_20[2]
        unbind_20 = None
        k_softmax_20 = k_20.softmax(dim=2)
        k_20 = None
        transpose_126 = k_softmax_20.transpose(-1, -2)
        k_softmax_20 = None
        factor_att_40 = transpose_126 @ v_20
        transpose_126 = None
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
        transpose_127 = v_img_40.transpose(-1, -2)
        v_img_40 = None
        v_img_41 = transpose_127.reshape(1, 512, 7, 7)
        transpose_127 = None
        split_20 = torch.functional.split(v_img_41, [128, 192, 192], dim=1)
        v_img_41 = None
        getitem_218 = split_20[0]
        getitem_219 = split_20[1]
        getitem_220 = split_20[2]
        split_20 = None
        conv2d_85 = torch.conv2d(
            getitem_218,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        getitem_218 = None
        conv2d_86 = torch.conv2d(
            getitem_219,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        getitem_219 = None
        conv2d_87 = torch.conv2d(
            getitem_220,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        getitem_220 = None
        conv_v_img_40 = torch.cat([conv2d_85, conv2d_86, conv2d_87], dim=1)
        conv2d_85 = conv2d_86 = conv2d_87 = None
        reshape_85 = conv_v_img_40.reshape(1, 8, 64, 49)
        conv_v_img_40 = None
        conv_v_img_41 = reshape_85.transpose(-1, -2)
        reshape_85 = None
        EV_hat_40 = q_img_20 * conv_v_img_41
        q_img_20 = conv_v_img_41 = None
        EV_hat_41 = torch._C._nn.pad(EV_hat_40, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_40 = None
        mul_41 = 0.125 * factor_att_41
        factor_att_41 = None
        x_340 = mul_41 + EV_hat_41
        mul_41 = EV_hat_41 = None
        transpose_129 = x_340.transpose(1, 2)
        x_340 = None
        x_341 = transpose_129.reshape(1, 50, 512)
        transpose_129 = None
        x_342 = torch._C._nn.linear(
            x_341,
            l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_341 = l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_1_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_343 = torch.nn.functional.dropout(x_342, 0.0, False, False)
        x_342 = None
        x_344 = x_338 + x_343
        x_338 = x_343 = None
        x_345 = torch.nn.functional.layer_norm(
            x_344,
            (512,),
            l_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_346 = torch._C._nn.linear(
            x_345,
            l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_345 = l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_347 = torch._C._nn.gelu(x_346, approximate="none")
        x_346 = None
        x_348 = torch.nn.functional.dropout(x_347, 0.0, False, False)
        x_347 = None
        x_349 = torch._C._nn.linear(
            x_348,
            l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_348 = l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks4_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_350 = torch.nn.functional.dropout(x_349, 0.0, False, False)
        x_349 = None
        x_351 = x_344 + x_350
        x_344 = x_350 = None
        cls_token_21 = x_351[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_21 = x_351[(slice(None, None, None), slice(1, None, None))]
        x_351 = None
        transpose_130 = img_tokens_21.transpose(1, 2)
        img_tokens_21 = None
        feat_21 = transpose_130.view(1, 512, 7, 7)
        transpose_130 = None
        conv2d_88 = torch.conv2d(
            feat_21,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_352 = conv2d_88 + feat_21
        conv2d_88 = feat_21 = None
        flatten_25 = x_352.flatten(2)
        x_352 = None
        x_353 = flatten_25.transpose(1, 2)
        flatten_25 = None
        x_354 = torch.cat((cls_token_21, x_353), dim=1)
        cls_token_21 = x_353 = None
        x_355 = torch.nn.functional.layer_norm(
            x_354,
            (512,),
            l_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_2_modules_norm1_parameters_bias_
        ) = None
        linear_84 = torch._C._nn.linear(
            x_355,
            l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_355 = l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_87 = linear_84.reshape(1, 50, 3, 8, 64)
        linear_84 = None
        qkv_21 = reshape_87.permute(2, 0, 3, 1, 4)
        reshape_87 = None
        unbind_21 = qkv_21.unbind(0)
        qkv_21 = None
        q_21 = unbind_21[0]
        k_21 = unbind_21[1]
        v_21 = unbind_21[2]
        unbind_21 = None
        k_softmax_21 = k_21.softmax(dim=2)
        k_21 = None
        transpose_132 = k_softmax_21.transpose(-1, -2)
        k_softmax_21 = None
        factor_att_42 = transpose_132 @ v_21
        transpose_132 = None
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
        transpose_133 = v_img_42.transpose(-1, -2)
        v_img_42 = None
        v_img_43 = transpose_133.reshape(1, 512, 7, 7)
        transpose_133 = None
        split_21 = torch.functional.split(v_img_43, [128, 192, 192], dim=1)
        v_img_43 = None
        getitem_228 = split_21[0]
        getitem_229 = split_21[1]
        getitem_230 = split_21[2]
        split_21 = None
        conv2d_89 = torch.conv2d(
            getitem_228,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        getitem_228 = None
        conv2d_90 = torch.conv2d(
            getitem_229,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        getitem_229 = None
        conv2d_91 = torch.conv2d(
            getitem_230,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        getitem_230 = None
        conv_v_img_42 = torch.cat([conv2d_89, conv2d_90, conv2d_91], dim=1)
        conv2d_89 = conv2d_90 = conv2d_91 = None
        reshape_89 = conv_v_img_42.reshape(1, 8, 64, 49)
        conv_v_img_42 = None
        conv_v_img_43 = reshape_89.transpose(-1, -2)
        reshape_89 = None
        EV_hat_42 = q_img_21 * conv_v_img_43
        q_img_21 = conv_v_img_43 = None
        EV_hat_43 = torch._C._nn.pad(EV_hat_42, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_42 = None
        mul_43 = 0.125 * factor_att_43
        factor_att_43 = None
        x_356 = mul_43 + EV_hat_43
        mul_43 = EV_hat_43 = None
        transpose_135 = x_356.transpose(1, 2)
        x_356 = None
        x_357 = transpose_135.reshape(1, 50, 512)
        transpose_135 = None
        x_358 = torch._C._nn.linear(
            x_357,
            l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_357 = l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_2_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_359 = torch.nn.functional.dropout(x_358, 0.0, False, False)
        x_358 = None
        x_360 = x_354 + x_359
        x_354 = x_359 = None
        x_361 = torch.nn.functional.layer_norm(
            x_360,
            (512,),
            l_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_362 = torch._C._nn.linear(
            x_361,
            l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_361 = l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_363 = torch._C._nn.gelu(x_362, approximate="none")
        x_362 = None
        x_364 = torch.nn.functional.dropout(x_363, 0.0, False, False)
        x_363 = None
        x_365 = torch._C._nn.linear(
            x_364,
            l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_364 = l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks4_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_366 = torch.nn.functional.dropout(x_365, 0.0, False, False)
        x_365 = None
        x_367 = x_360 + x_366
        x_360 = x_366 = None
        cls_token_22 = x_367[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_22 = x_367[(slice(None, None, None), slice(1, None, None))]
        x_367 = None
        transpose_136 = img_tokens_22.transpose(1, 2)
        img_tokens_22 = None
        feat_22 = transpose_136.view(1, 512, 7, 7)
        transpose_136 = None
        conv2d_92 = torch.conv2d(
            feat_22,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_368 = conv2d_92 + feat_22
        conv2d_92 = feat_22 = None
        flatten_26 = x_368.flatten(2)
        x_368 = None
        x_369 = flatten_26.transpose(1, 2)
        flatten_26 = None
        x_370 = torch.cat((cls_token_22, x_369), dim=1)
        cls_token_22 = x_369 = None
        x_371 = torch.nn.functional.layer_norm(
            x_370,
            (512,),
            l_self_modules_serial_blocks4_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_3_modules_norm1_parameters_bias_
        ) = None
        linear_88 = torch._C._nn.linear(
            x_371,
            l_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_371 = l_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_91 = linear_88.reshape(1, 50, 3, 8, 64)
        linear_88 = None
        qkv_22 = reshape_91.permute(2, 0, 3, 1, 4)
        reshape_91 = None
        unbind_22 = qkv_22.unbind(0)
        qkv_22 = None
        q_22 = unbind_22[0]
        k_22 = unbind_22[1]
        v_22 = unbind_22[2]
        unbind_22 = None
        k_softmax_22 = k_22.softmax(dim=2)
        k_22 = None
        transpose_138 = k_softmax_22.transpose(-1, -2)
        k_softmax_22 = None
        factor_att_44 = transpose_138 @ v_22
        transpose_138 = None
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
        transpose_139 = v_img_44.transpose(-1, -2)
        v_img_44 = None
        v_img_45 = transpose_139.reshape(1, 512, 7, 7)
        transpose_139 = None
        split_22 = torch.functional.split(v_img_45, [128, 192, 192], dim=1)
        v_img_45 = None
        getitem_238 = split_22[0]
        getitem_239 = split_22[1]
        getitem_240 = split_22[2]
        split_22 = None
        conv2d_93 = torch.conv2d(
            getitem_238,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        getitem_238 = None
        conv2d_94 = torch.conv2d(
            getitem_239,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        getitem_239 = None
        conv2d_95 = torch.conv2d(
            getitem_240,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        getitem_240 = None
        conv_v_img_44 = torch.cat([conv2d_93, conv2d_94, conv2d_95], dim=1)
        conv2d_93 = conv2d_94 = conv2d_95 = None
        reshape_93 = conv_v_img_44.reshape(1, 8, 64, 49)
        conv_v_img_44 = None
        conv_v_img_45 = reshape_93.transpose(-1, -2)
        reshape_93 = None
        EV_hat_44 = q_img_22 * conv_v_img_45
        q_img_22 = conv_v_img_45 = None
        EV_hat_45 = torch._C._nn.pad(EV_hat_44, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_44 = None
        mul_45 = 0.125 * factor_att_45
        factor_att_45 = None
        x_372 = mul_45 + EV_hat_45
        mul_45 = EV_hat_45 = None
        transpose_141 = x_372.transpose(1, 2)
        x_372 = None
        x_373 = transpose_141.reshape(1, 50, 512)
        transpose_141 = None
        x_374 = torch._C._nn.linear(
            x_373,
            l_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_373 = l_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_3_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_375 = torch.nn.functional.dropout(x_374, 0.0, False, False)
        x_374 = None
        x_376 = x_370 + x_375
        x_370 = x_375 = None
        x_377 = torch.nn.functional.layer_norm(
            x_376,
            (512,),
            l_self_modules_serial_blocks4_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_378 = torch._C._nn.linear(
            x_377,
            l_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_377 = l_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_379 = torch._C._nn.gelu(x_378, approximate="none")
        x_378 = None
        x_380 = torch.nn.functional.dropout(x_379, 0.0, False, False)
        x_379 = None
        x_381 = torch._C._nn.linear(
            x_380,
            l_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_380 = l_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks4_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_382 = torch.nn.functional.dropout(x_381, 0.0, False, False)
        x_381 = None
        x_383 = x_376 + x_382
        x_376 = x_382 = None
        cls_token_23 = x_383[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_23 = x_383[(slice(None, None, None), slice(1, None, None))]
        x_383 = None
        transpose_142 = img_tokens_23.transpose(1, 2)
        img_tokens_23 = None
        feat_23 = transpose_142.view(1, 512, 7, 7)
        transpose_142 = None
        conv2d_96 = torch.conv2d(
            feat_23,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_384 = conv2d_96 + feat_23
        conv2d_96 = feat_23 = None
        flatten_27 = x_384.flatten(2)
        x_384 = None
        x_385 = flatten_27.transpose(1, 2)
        flatten_27 = None
        x_386 = torch.cat((cls_token_23, x_385), dim=1)
        cls_token_23 = x_385 = None
        x_387 = torch.nn.functional.layer_norm(
            x_386,
            (512,),
            l_self_modules_serial_blocks4_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_4_modules_norm1_parameters_bias_
        ) = None
        linear_92 = torch._C._nn.linear(
            x_387,
            l_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_387 = l_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_95 = linear_92.reshape(1, 50, 3, 8, 64)
        linear_92 = None
        qkv_23 = reshape_95.permute(2, 0, 3, 1, 4)
        reshape_95 = None
        unbind_23 = qkv_23.unbind(0)
        qkv_23 = None
        q_23 = unbind_23[0]
        k_23 = unbind_23[1]
        v_23 = unbind_23[2]
        unbind_23 = None
        k_softmax_23 = k_23.softmax(dim=2)
        k_23 = None
        transpose_144 = k_softmax_23.transpose(-1, -2)
        k_softmax_23 = None
        factor_att_46 = transpose_144 @ v_23
        transpose_144 = None
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
        transpose_145 = v_img_46.transpose(-1, -2)
        v_img_46 = None
        v_img_47 = transpose_145.reshape(1, 512, 7, 7)
        transpose_145 = None
        split_23 = torch.functional.split(v_img_47, [128, 192, 192], dim=1)
        v_img_47 = None
        getitem_248 = split_23[0]
        getitem_249 = split_23[1]
        getitem_250 = split_23[2]
        split_23 = None
        conv2d_97 = torch.conv2d(
            getitem_248,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        getitem_248 = None
        conv2d_98 = torch.conv2d(
            getitem_249,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        getitem_249 = None
        conv2d_99 = torch.conv2d(
            getitem_250,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        getitem_250 = None
        conv_v_img_46 = torch.cat([conv2d_97, conv2d_98, conv2d_99], dim=1)
        conv2d_97 = conv2d_98 = conv2d_99 = None
        reshape_97 = conv_v_img_46.reshape(1, 8, 64, 49)
        conv_v_img_46 = None
        conv_v_img_47 = reshape_97.transpose(-1, -2)
        reshape_97 = None
        EV_hat_46 = q_img_23 * conv_v_img_47
        q_img_23 = conv_v_img_47 = None
        EV_hat_47 = torch._C._nn.pad(EV_hat_46, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_46 = None
        mul_47 = 0.125 * factor_att_47
        factor_att_47 = None
        x_388 = mul_47 + EV_hat_47
        mul_47 = EV_hat_47 = None
        transpose_147 = x_388.transpose(1, 2)
        x_388 = None
        x_389 = transpose_147.reshape(1, 50, 512)
        transpose_147 = None
        x_390 = torch._C._nn.linear(
            x_389,
            l_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_389 = l_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_4_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_391 = torch.nn.functional.dropout(x_390, 0.0, False, False)
        x_390 = None
        x_392 = x_386 + x_391
        x_386 = x_391 = None
        x_393 = torch.nn.functional.layer_norm(
            x_392,
            (512,),
            l_self_modules_serial_blocks4_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_394 = torch._C._nn.linear(
            x_393,
            l_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_393 = l_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_395 = torch._C._nn.gelu(x_394, approximate="none")
        x_394 = None
        x_396 = torch.nn.functional.dropout(x_395, 0.0, False, False)
        x_395 = None
        x_397 = torch._C._nn.linear(
            x_396,
            l_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_396 = l_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks4_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_398 = torch.nn.functional.dropout(x_397, 0.0, False, False)
        x_397 = None
        x_399 = x_392 + x_398
        x_392 = x_398 = None
        cls_token_24 = x_399[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_24 = x_399[(slice(None, None, None), slice(1, None, None))]
        x_399 = None
        transpose_148 = img_tokens_24.transpose(1, 2)
        img_tokens_24 = None
        feat_24 = transpose_148.view(1, 512, 7, 7)
        transpose_148 = None
        conv2d_100 = torch.conv2d(
            feat_24,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_400 = conv2d_100 + feat_24
        conv2d_100 = feat_24 = None
        flatten_28 = x_400.flatten(2)
        x_400 = None
        x_401 = flatten_28.transpose(1, 2)
        flatten_28 = None
        x_402 = torch.cat((cls_token_24, x_401), dim=1)
        cls_token_24 = x_401 = None
        x_403 = torch.nn.functional.layer_norm(
            x_402,
            (512,),
            l_self_modules_serial_blocks4_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_5_modules_norm1_parameters_bias_
        ) = None
        linear_96 = torch._C._nn.linear(
            x_403,
            l_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_403 = l_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_99 = linear_96.reshape(1, 50, 3, 8, 64)
        linear_96 = None
        qkv_24 = reshape_99.permute(2, 0, 3, 1, 4)
        reshape_99 = None
        unbind_24 = qkv_24.unbind(0)
        qkv_24 = None
        q_24 = unbind_24[0]
        k_24 = unbind_24[1]
        v_24 = unbind_24[2]
        unbind_24 = None
        k_softmax_24 = k_24.softmax(dim=2)
        k_24 = None
        transpose_150 = k_softmax_24.transpose(-1, -2)
        k_softmax_24 = None
        factor_att_48 = transpose_150 @ v_24
        transpose_150 = None
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
        transpose_151 = v_img_48.transpose(-1, -2)
        v_img_48 = None
        v_img_49 = transpose_151.reshape(1, 512, 7, 7)
        transpose_151 = None
        split_24 = torch.functional.split(v_img_49, [128, 192, 192], dim=1)
        v_img_49 = None
        getitem_258 = split_24[0]
        getitem_259 = split_24[1]
        getitem_260 = split_24[2]
        split_24 = None
        conv2d_101 = torch.conv2d(
            getitem_258,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        getitem_258 = None
        conv2d_102 = torch.conv2d(
            getitem_259,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        getitem_259 = None
        conv2d_103 = torch.conv2d(
            getitem_260,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        getitem_260 = None
        conv_v_img_48 = torch.cat([conv2d_101, conv2d_102, conv2d_103], dim=1)
        conv2d_101 = conv2d_102 = conv2d_103 = None
        reshape_101 = conv_v_img_48.reshape(1, 8, 64, 49)
        conv_v_img_48 = None
        conv_v_img_49 = reshape_101.transpose(-1, -2)
        reshape_101 = None
        EV_hat_48 = q_img_24 * conv_v_img_49
        q_img_24 = conv_v_img_49 = None
        EV_hat_49 = torch._C._nn.pad(EV_hat_48, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_48 = None
        mul_49 = 0.125 * factor_att_49
        factor_att_49 = None
        x_404 = mul_49 + EV_hat_49
        mul_49 = EV_hat_49 = None
        transpose_153 = x_404.transpose(1, 2)
        x_404 = None
        x_405 = transpose_153.reshape(1, 50, 512)
        transpose_153 = None
        x_406 = torch._C._nn.linear(
            x_405,
            l_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_405 = l_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_5_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_407 = torch.nn.functional.dropout(x_406, 0.0, False, False)
        x_406 = None
        x_408 = x_402 + x_407
        x_402 = x_407 = None
        x_409 = torch.nn.functional.layer_norm(
            x_408,
            (512,),
            l_self_modules_serial_blocks4_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_410 = torch._C._nn.linear(
            x_409,
            l_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_409 = l_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_411 = torch._C._nn.gelu(x_410, approximate="none")
        x_410 = None
        x_412 = torch.nn.functional.dropout(x_411, 0.0, False, False)
        x_411 = None
        x_413 = torch._C._nn.linear(
            x_412,
            l_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_412 = l_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks4_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_414 = torch.nn.functional.dropout(x_413, 0.0, False, False)
        x_413 = None
        x_415 = x_408 + x_414
        x_408 = x_414 = None
        cls_token_25 = x_415[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_25 = x_415[(slice(None, None, None), slice(1, None, None))]
        x_415 = None
        transpose_154 = img_tokens_25.transpose(1, 2)
        img_tokens_25 = None
        feat_25 = transpose_154.view(1, 512, 7, 7)
        transpose_154 = None
        conv2d_104 = torch.conv2d(
            feat_25,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_416 = conv2d_104 + feat_25
        conv2d_104 = feat_25 = None
        flatten_29 = x_416.flatten(2)
        x_416 = None
        x_417 = flatten_29.transpose(1, 2)
        flatten_29 = None
        x_418 = torch.cat((cls_token_25, x_417), dim=1)
        cls_token_25 = x_417 = None
        x_419 = torch.nn.functional.layer_norm(
            x_418,
            (512,),
            l_self_modules_serial_blocks4_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_6_modules_norm1_parameters_bias_
        ) = None
        linear_100 = torch._C._nn.linear(
            x_419,
            l_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_419 = l_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_103 = linear_100.reshape(1, 50, 3, 8, 64)
        linear_100 = None
        qkv_25 = reshape_103.permute(2, 0, 3, 1, 4)
        reshape_103 = None
        unbind_25 = qkv_25.unbind(0)
        qkv_25 = None
        q_25 = unbind_25[0]
        k_25 = unbind_25[1]
        v_25 = unbind_25[2]
        unbind_25 = None
        k_softmax_25 = k_25.softmax(dim=2)
        k_25 = None
        transpose_156 = k_softmax_25.transpose(-1, -2)
        k_softmax_25 = None
        factor_att_50 = transpose_156 @ v_25
        transpose_156 = None
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
        transpose_157 = v_img_50.transpose(-1, -2)
        v_img_50 = None
        v_img_51 = transpose_157.reshape(1, 512, 7, 7)
        transpose_157 = None
        split_25 = torch.functional.split(v_img_51, [128, 192, 192], dim=1)
        v_img_51 = None
        getitem_268 = split_25[0]
        getitem_269 = split_25[1]
        getitem_270 = split_25[2]
        split_25 = None
        conv2d_105 = torch.conv2d(
            getitem_268,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        getitem_268 = None
        conv2d_106 = torch.conv2d(
            getitem_269,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        getitem_269 = None
        conv2d_107 = torch.conv2d(
            getitem_270,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        getitem_270 = None
        conv_v_img_50 = torch.cat([conv2d_105, conv2d_106, conv2d_107], dim=1)
        conv2d_105 = conv2d_106 = conv2d_107 = None
        reshape_105 = conv_v_img_50.reshape(1, 8, 64, 49)
        conv_v_img_50 = None
        conv_v_img_51 = reshape_105.transpose(-1, -2)
        reshape_105 = None
        EV_hat_50 = q_img_25 * conv_v_img_51
        q_img_25 = conv_v_img_51 = None
        EV_hat_51 = torch._C._nn.pad(EV_hat_50, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_50 = None
        mul_51 = 0.125 * factor_att_51
        factor_att_51 = None
        x_420 = mul_51 + EV_hat_51
        mul_51 = EV_hat_51 = None
        transpose_159 = x_420.transpose(1, 2)
        x_420 = None
        x_421 = transpose_159.reshape(1, 50, 512)
        transpose_159 = None
        x_422 = torch._C._nn.linear(
            x_421,
            l_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_421 = l_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_6_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_423 = torch.nn.functional.dropout(x_422, 0.0, False, False)
        x_422 = None
        x_424 = x_418 + x_423
        x_418 = x_423 = None
        x_425 = torch.nn.functional.layer_norm(
            x_424,
            (512,),
            l_self_modules_serial_blocks4_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_6_modules_norm2_parameters_bias_
        ) = None
        x_426 = torch._C._nn.linear(
            x_425,
            l_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_425 = l_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_427 = torch._C._nn.gelu(x_426, approximate="none")
        x_426 = None
        x_428 = torch.nn.functional.dropout(x_427, 0.0, False, False)
        x_427 = None
        x_429 = torch._C._nn.linear(
            x_428,
            l_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_428 = l_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks4_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_430 = torch.nn.functional.dropout(x_429, 0.0, False, False)
        x_429 = None
        x_431 = x_424 + x_430
        x_424 = x_430 = None
        cls_token_26 = x_431[(slice(None, None, None), slice(None, 1, None))]
        img_tokens_26 = x_431[(slice(None, None, None), slice(1, None, None))]
        x_431 = None
        transpose_160 = img_tokens_26.transpose(1, 2)
        img_tokens_26 = None
        feat_26 = transpose_160.view(1, 512, 7, 7)
        transpose_160 = None
        conv2d_108 = torch.conv2d(
            feat_26,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_ = (None)
        x_432 = conv2d_108 + feat_26
        conv2d_108 = feat_26 = None
        flatten_30 = x_432.flatten(2)
        x_432 = None
        x_433 = flatten_30.transpose(1, 2)
        flatten_30 = None
        x_434 = torch.cat((cls_token_26, x_433), dim=1)
        cls_token_26 = x_433 = None
        x_435 = torch.nn.functional.layer_norm(
            x_434,
            (512,),
            l_self_modules_serial_blocks4_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_7_modules_norm1_parameters_bias_
        ) = None
        linear_104 = torch._C._nn.linear(
            x_435,
            l_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_qkv_parameters_weight_,
            l_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_qkv_parameters_bias_,
        )
        x_435 = l_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_qkv_parameters_weight_ = l_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_qkv_parameters_bias_ = (None)
        reshape_107 = linear_104.reshape(1, 50, 3, 8, 64)
        linear_104 = None
        qkv_26 = reshape_107.permute(2, 0, 3, 1, 4)
        reshape_107 = None
        unbind_26 = qkv_26.unbind(0)
        qkv_26 = None
        q_26 = unbind_26[0]
        k_26 = unbind_26[1]
        v_26 = unbind_26[2]
        unbind_26 = None
        k_softmax_26 = k_26.softmax(dim=2)
        k_26 = None
        transpose_162 = k_softmax_26.transpose(-1, -2)
        k_softmax_26 = None
        factor_att_52 = transpose_162 @ v_26
        transpose_162 = None
        factor_att_53 = q_26 @ factor_att_52
        factor_att_52 = None
        q_img_26 = q_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_26 = None
        v_img_52 = v_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        v_26 = None
        transpose_163 = v_img_52.transpose(-1, -2)
        v_img_52 = None
        v_img_53 = transpose_163.reshape(1, 512, 7, 7)
        transpose_163 = None
        split_26 = torch.functional.split(v_img_53, [128, 192, 192], dim=1)
        v_img_53 = None
        getitem_278 = split_26[0]
        getitem_279 = split_26[1]
        getitem_280 = split_26[2]
        split_26 = None
        conv2d_109 = torch.conv2d(
            getitem_278,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        getitem_278 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_110 = torch.conv2d(
            getitem_279,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        getitem_279 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_111 = torch.conv2d(
            getitem_280,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        getitem_280 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
        conv_v_img_52 = torch.cat([conv2d_109, conv2d_110, conv2d_111], dim=1)
        conv2d_109 = conv2d_110 = conv2d_111 = None
        reshape_109 = conv_v_img_52.reshape(1, 8, 64, 49)
        conv_v_img_52 = None
        conv_v_img_53 = reshape_109.transpose(-1, -2)
        reshape_109 = None
        EV_hat_52 = q_img_26 * conv_v_img_53
        q_img_26 = conv_v_img_53 = None
        EV_hat_53 = torch._C._nn.pad(EV_hat_52, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_52 = None
        mul_53 = 0.125 * factor_att_53
        factor_att_53 = None
        x_436 = mul_53 + EV_hat_53
        mul_53 = EV_hat_53 = None
        transpose_165 = x_436.transpose(1, 2)
        x_436 = None
        x_437 = transpose_165.reshape(1, 50, 512)
        transpose_165 = None
        x_438 = torch._C._nn.linear(
            x_437,
            l_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_proj_parameters_bias_,
        )
        x_437 = l_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_7_modules_factoratt_crpe_modules_proj_parameters_bias_ = (None)
        x_439 = torch.nn.functional.dropout(x_438, 0.0, False, False)
        x_438 = None
        x_440 = x_434 + x_439
        x_434 = x_439 = None
        x_441 = torch.nn.functional.layer_norm(
            x_440,
            (512,),
            l_self_modules_serial_blocks4_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_serial_blocks4_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_serial_blocks4_modules_7_modules_norm2_parameters_bias_
        ) = None
        x_442 = torch._C._nn.linear(
            x_441,
            l_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_441 = l_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_443 = torch._C._nn.gelu(x_442, approximate="none")
        x_442 = None
        x_444 = torch.nn.functional.dropout(x_443, 0.0, False, False)
        x_443 = None
        x_445 = torch._C._nn.linear(
            x_444,
            l_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_444 = l_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_serial_blocks4_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_446 = torch.nn.functional.dropout(x_445, 0.0, False, False)
        x_445 = None
        x_447 = x_440 + x_446
        x_440 = x_446 = None
        getitem_281 = x_447[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        reshape_111 = getitem_281.reshape(1, 7, 7, -1)
        getitem_281 = None
        permute_30 = reshape_111.permute(0, 3, 1, 2)
        reshape_111 = None
        x4_nocls = permute_30.contiguous()
        permute_30 = x4_nocls = None
        x_448 = torch.nn.functional.layer_norm(
            x_447,
            (512,),
            l_self_modules_norm4_parameters_weight_,
            l_self_modules_norm4_parameters_bias_,
            1e-06,
        )
        x_447 = (
            l_self_modules_norm4_parameters_weight_
        ) = l_self_modules_norm4_parameters_bias_ = None
        x_449 = x_448[(slice(None, None, None), 0)]
        x_448 = None
        x_450 = torch.nn.functional.dropout(x_449, 0.0, False, False)
        x_449 = None
        x_451 = torch._C._nn.linear(
            x_450,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_450 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_451,)
