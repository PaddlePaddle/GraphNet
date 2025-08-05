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
        getitem_25 = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_7 = torch.conv2d(
            getitem_26,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            24,
        )
        getitem_26 = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_8 = torch.conv2d(
            getitem_27,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            24,
        )
        getitem_27 = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks1_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
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
        getitem_28 = x_35[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        x_35 = None
        reshape_8 = getitem_28.reshape(1, 56, 56, -1)
        getitem_28 = None
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
            (128,),
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
        feat_2 = transpose_14.view(1, 128, 28, 28)
        transpose_14 = None
        conv2d_10 = torch.conv2d(
            feat_2,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
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
            (128,),
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
        reshape_9 = linear_8.reshape(1, 785, 3, 8, 16)
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
        v_img_5 = transpose_17.reshape(1, 128, 28, 28)
        transpose_17 = None
        split_2 = torch.functional.split(v_img_5, [32, 48, 48], dim=1)
        v_img_5 = None
        getitem_36 = split_2[0]
        getitem_37 = split_2[1]
        getitem_38 = split_2[2]
        split_2 = None
        conv2d_11 = torch.conv2d(
            getitem_36,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        getitem_36 = None
        conv2d_12 = torch.conv2d(
            getitem_37,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            48,
        )
        getitem_37 = None
        conv2d_13 = torch.conv2d(
            getitem_38,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            48,
        )
        getitem_38 = None
        conv_v_img_4 = torch.cat([conv2d_11, conv2d_12, conv2d_13], dim=1)
        conv2d_11 = conv2d_12 = conv2d_13 = None
        reshape_11 = conv_v_img_4.reshape(1, 8, 16, 784)
        conv_v_img_4 = None
        conv_v_img_5 = reshape_11.transpose(-1, -2)
        reshape_11 = None
        EV_hat_4 = q_img_2 * conv_v_img_5
        q_img_2 = conv_v_img_5 = None
        EV_hat_5 = torch._C._nn.pad(EV_hat_4, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_4 = None
        mul_5 = 0.25 * factor_att_5
        factor_att_5 = None
        x_44 = mul_5 + EV_hat_5
        mul_5 = EV_hat_5 = None
        transpose_19 = x_44.transpose(1, 2)
        x_44 = None
        x_45 = transpose_19.reshape(1, 785, 128)
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
            (128,),
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
        l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_cpe_modules_proj_parameters_bias_ = (None)
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
        getitem_46 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_16 = torch.conv2d(
            getitem_47,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            48,
        )
        getitem_47 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_17 = torch.conv2d(
            getitem_48,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            48,
        )
        getitem_48 = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks2_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
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
            (128,),
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
        getitem_49 = x_71[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        x_71 = None
        reshape_17 = getitem_49.reshape(1, 28, 28, -1)
        getitem_49 = None
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
            (320,),
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
        feat_4 = transpose_27.view(1, 320, 14, 14)
        transpose_27 = None
        conv2d_19 = torch.conv2d(
            feat_4,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
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
            (320,),
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
        reshape_18 = linear_16.reshape(1, 197, 3, 8, 40)
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
        v_img_9 = transpose_30.reshape(1, 320, 14, 14)
        transpose_30 = None
        split_4 = torch.functional.split(v_img_9, [80, 120, 120], dim=1)
        v_img_9 = None
        getitem_57 = split_4[0]
        getitem_58 = split_4[1]
        getitem_59 = split_4[2]
        split_4 = None
        conv2d_20 = torch.conv2d(
            getitem_57,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        getitem_57 = None
        conv2d_21 = torch.conv2d(
            getitem_58,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_58 = None
        conv2d_22 = torch.conv2d(
            getitem_59,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_59 = None
        conv_v_img_8 = torch.cat([conv2d_20, conv2d_21, conv2d_22], dim=1)
        conv2d_20 = conv2d_21 = conv2d_22 = None
        reshape_20 = conv_v_img_8.reshape(1, 8, 40, 196)
        conv_v_img_8 = None
        conv_v_img_9 = reshape_20.transpose(-1, -2)
        reshape_20 = None
        EV_hat_8 = q_img_4 * conv_v_img_9
        q_img_4 = conv_v_img_9 = None
        EV_hat_9 = torch._C._nn.pad(EV_hat_8, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_8 = None
        mul_9 = 0.15811388300841897 * factor_att_9
        factor_att_9 = None
        x_80 = mul_9 + EV_hat_9
        mul_9 = EV_hat_9 = None
        transpose_32 = x_80.transpose(1, 2)
        x_80 = None
        x_81 = transpose_32.reshape(1, 197, 320)
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
            (320,),
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
        feat_5 = transpose_33.view(1, 320, 14, 14)
        transpose_33 = None
        conv2d_23 = torch.conv2d(
            feat_5,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_cpe_modules_proj_parameters_bias_ = (None)
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
            (320,),
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
        reshape_22 = linear_20.reshape(1, 197, 3, 8, 40)
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
        v_img_11 = transpose_36.reshape(1, 320, 14, 14)
        transpose_36 = None
        split_5 = torch.functional.split(v_img_11, [80, 120, 120], dim=1)
        v_img_11 = None
        getitem_67 = split_5[0]
        getitem_68 = split_5[1]
        getitem_69 = split_5[2]
        split_5 = None
        conv2d_24 = torch.conv2d(
            getitem_67,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        getitem_67 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_25 = torch.conv2d(
            getitem_68,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_68 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_26 = torch.conv2d(
            getitem_69,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_69 = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks3_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
        conv_v_img_10 = torch.cat([conv2d_24, conv2d_25, conv2d_26], dim=1)
        conv2d_24 = conv2d_25 = conv2d_26 = None
        reshape_24 = conv_v_img_10.reshape(1, 8, 40, 196)
        conv_v_img_10 = None
        conv_v_img_11 = reshape_24.transpose(-1, -2)
        reshape_24 = None
        EV_hat_10 = q_img_5 * conv_v_img_11
        q_img_5 = conv_v_img_11 = None
        EV_hat_11 = torch._C._nn.pad(EV_hat_10, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_10 = None
        mul_11 = 0.15811388300841897 * factor_att_11
        factor_att_11 = None
        x_96 = mul_11 + EV_hat_11
        mul_11 = EV_hat_11 = None
        transpose_38 = x_96.transpose(1, 2)
        x_96 = None
        x_97 = transpose_38.reshape(1, 197, 320)
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
            (320,),
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
        getitem_70 = x_107[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        x_107 = None
        reshape_26 = getitem_70.reshape(1, 14, 14, -1)
        getitem_70 = None
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
            (512,),
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
        feat_6 = transpose_40.view(1, 512, 7, 7)
        transpose_40 = None
        conv2d_28 = torch.conv2d(
            feat_6,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
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
            (512,),
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
        reshape_27 = linear_24.reshape(1, 50, 3, 8, 64)
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
        v_img_13 = transpose_43.reshape(1, 512, 7, 7)
        transpose_43 = None
        split_6 = torch.functional.split(v_img_13, [128, 192, 192], dim=1)
        v_img_13 = None
        getitem_78 = split_6[0]
        getitem_79 = split_6[1]
        getitem_80 = split_6[2]
        split_6 = None
        conv2d_29 = torch.conv2d(
            getitem_78,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        getitem_78 = None
        conv2d_30 = torch.conv2d(
            getitem_79,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        getitem_79 = None
        conv2d_31 = torch.conv2d(
            getitem_80,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        getitem_80 = None
        conv_v_img_12 = torch.cat([conv2d_29, conv2d_30, conv2d_31], dim=1)
        conv2d_29 = conv2d_30 = conv2d_31 = None
        reshape_29 = conv_v_img_12.reshape(1, 8, 64, 49)
        conv_v_img_12 = None
        conv_v_img_13 = reshape_29.transpose(-1, -2)
        reshape_29 = None
        EV_hat_12 = q_img_6 * conv_v_img_13
        q_img_6 = conv_v_img_13 = None
        EV_hat_13 = torch._C._nn.pad(EV_hat_12, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_12 = None
        mul_13 = 0.125 * factor_att_13
        factor_att_13 = None
        x_116 = mul_13 + EV_hat_13
        mul_13 = EV_hat_13 = None
        transpose_45 = x_116.transpose(1, 2)
        x_116 = None
        x_117 = transpose_45.reshape(1, 50, 512)
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
            (512,),
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
        feat_7 = transpose_46.view(1, 512, 7, 7)
        transpose_46 = None
        conv2d_32 = torch.conv2d(
            feat_7,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_cpe_modules_proj_parameters_bias_ = (None)
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
            (512,),
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
        reshape_31 = linear_28.reshape(1, 50, 3, 8, 64)
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
        v_img_15 = transpose_49.reshape(1, 512, 7, 7)
        transpose_49 = None
        split_7 = torch.functional.split(v_img_15, [128, 192, 192], dim=1)
        v_img_15 = None
        getitem_88 = split_7[0]
        getitem_89 = split_7[1]
        getitem_90 = split_7[2]
        split_7 = None
        conv2d_33 = torch.conv2d(
            getitem_88,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        getitem_88 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_0_parameters_bias_ = (None)
        conv2d_34 = torch.conv2d(
            getitem_89,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        getitem_89 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_1_parameters_bias_ = (None)
        conv2d_35 = torch.conv2d(
            getitem_90,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_,
            l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        getitem_90 = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_weight_ = l_self_modules_serial_blocks4_modules_0_modules_factoratt_crpe_modules_crpe_modules_conv_list_modules_2_parameters_bias_ = (None)
        conv_v_img_14 = torch.cat([conv2d_33, conv2d_34, conv2d_35], dim=1)
        conv2d_33 = conv2d_34 = conv2d_35 = None
        reshape_33 = conv_v_img_14.reshape(1, 8, 64, 49)
        conv_v_img_14 = None
        conv_v_img_15 = reshape_33.transpose(-1, -2)
        reshape_33 = None
        EV_hat_14 = q_img_7 * conv_v_img_15
        q_img_7 = conv_v_img_15 = None
        EV_hat_15 = torch._C._nn.pad(EV_hat_14, (0, 0, 1, 0, 0, 0), "constant", None)
        EV_hat_14 = None
        mul_15 = 0.125 * factor_att_15
        factor_att_15 = None
        x_132 = mul_15 + EV_hat_15
        mul_15 = EV_hat_15 = None
        transpose_51 = x_132.transpose(1, 2)
        x_132 = None
        x_133 = transpose_51.reshape(1, 50, 512)
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
            (512,),
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
        getitem_91 = x_143[
            (slice(None, None, None), slice(1, None, None), slice(None, None, None))
        ]
        reshape_35 = getitem_91.reshape(1, 7, 7, -1)
        getitem_91 = None
        permute_11 = reshape_35.permute(0, 3, 1, 2)
        reshape_35 = None
        x4_nocls = permute_11.contiguous()
        permute_11 = x4_nocls = None
        x_144 = torch.nn.functional.layer_norm(
            x_143,
            (512,),
            l_self_modules_norm4_parameters_weight_,
            l_self_modules_norm4_parameters_bias_,
            1e-06,
        )
        x_143 = (
            l_self_modules_norm4_parameters_weight_
        ) = l_self_modules_norm4_parameters_bias_ = None
        x_145 = x_144[(slice(None, None, None), 0)]
        x_144 = None
        x_146 = torch.nn.functional.dropout(x_145, 0.0, False, False)
        x_145 = None
        x_147 = torch._C._nn.linear(
            x_146,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_146 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_147,)
