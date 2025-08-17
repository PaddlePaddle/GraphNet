import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_parameters_pos_embed_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_1_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_2_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_3_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_3_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_convs_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_convs_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_project_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_project_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_project_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_project_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_project_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_project_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_project_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_project_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_project_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_project_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_project_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_project_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_project_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_ = L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_
        l_self_modules_backbone_parameters_cls_token_ = (
            L_self_modules_backbone_parameters_cls_token_
        )
        l_self_modules_backbone_parameters_pos_embed_ = (
            L_self_modules_backbone_parameters_pos_embed_
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_0_modules_0_parameters_weight_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_0_modules_0_parameters_weight_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_0_modules_0_parameters_bias_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_0_modules_0_parameters_bias_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_0_modules_conv_parameters_bias_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_0_modules_conv_parameters_bias_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_0_parameters_weight_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_0_parameters_weight_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_0_parameters_bias_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_0_parameters_bias_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_1_modules_0_parameters_weight_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_1_modules_0_parameters_weight_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_1_modules_0_parameters_bias_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_1_modules_0_parameters_bias_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_1_modules_conv_parameters_bias_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_1_modules_conv_parameters_bias_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_1_parameters_weight_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_1_parameters_weight_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_1_parameters_bias_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_1_parameters_bias_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_2_modules_0_parameters_weight_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_2_modules_0_parameters_weight_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_2_modules_0_parameters_bias_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_2_modules_0_parameters_bias_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_2_modules_conv_parameters_bias_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_2_modules_conv_parameters_bias_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_3_modules_0_parameters_weight_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_3_modules_0_parameters_weight_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_3_modules_0_parameters_bias_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_3_modules_0_parameters_bias_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_3_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_3_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_3_modules_conv_parameters_bias_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_3_modules_conv_parameters_bias_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_3_parameters_weight_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_3_parameters_weight_
        l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_3_parameters_bias_ = L_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_3_parameters_bias_
        l_self_modules_decode_head_modules_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_convs_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_convs_modules_2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_convs_modules_2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_convs_modules_3_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_convs_modules_3_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_project_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_project_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_project_modules_conv_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_project_modules_conv_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_project_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_project_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_project_modules_conv_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_project_modules_conv_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_project_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_project_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_project_modules_conv_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_project_modules_conv_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_project_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_project_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_project_modules_conv_parameters_bias_ = L_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_project_modules_conv_parameters_bias_
        l_self_modules_decode_head_modules_project_modules_conv_parameters_weight_ = (
            L_self_modules_decode_head_modules_project_modules_conv_parameters_weight_
        )
        l_self_modules_decode_head_modules_project_modules_bn_buffers_running_mean_ = (
            L_self_modules_decode_head_modules_project_modules_bn_buffers_running_mean_
        )
        l_self_modules_decode_head_modules_project_modules_bn_buffers_running_var_ = (
            L_self_modules_decode_head_modules_project_modules_bn_buffers_running_var_
        )
        l_self_modules_decode_head_modules_project_modules_bn_parameters_weight_ = (
            L_self_modules_decode_head_modules_project_modules_bn_parameters_weight_
        )
        l_self_modules_decode_head_modules_project_modules_bn_parameters_bias_ = (
            L_self_modules_decode_head_modules_project_modules_bn_parameters_bias_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_weight_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_bias_
        )
        x = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_,
            None,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_ = (None)
        flatten = x.flatten(2)
        x = None
        x_1 = flatten.transpose(1, 2)
        flatten = None
        cls_tokens = l_self_modules_backbone_parameters_cls_token_.expand(1, -1, -1)
        l_self_modules_backbone_parameters_cls_token_ = None
        x_2 = torch.cat((cls_tokens, x_1), dim=1)
        cls_tokens = x_1 = None
        cls_token_weight = l_self_modules_backbone_parameters_pos_embed_[
            (slice(None, None, None), 0)
        ]
        pos_embed_weight = l_self_modules_backbone_parameters_pos_embed_[
            (slice(None, None, None), slice(-196, None, None))
        ]
        l_self_modules_backbone_parameters_pos_embed_ = None
        reshape = pos_embed_weight.reshape(1, 14, 14, 768)
        pos_embed_weight = None
        pos_embed_weight_1 = reshape.permute(0, 3, 1, 2)
        reshape = None
        pos_embed_weight_2 = torch.nn.functional.interpolate(
            pos_embed_weight_1, (32, 32), None, "bicubic", False
        )
        pos_embed_weight_1 = None
        cls_token_weight_1 = cls_token_weight.unsqueeze(1)
        cls_token_weight = None
        flatten_1 = torch.flatten(pos_embed_weight_2, 2)
        pos_embed_weight_2 = None
        pos_embed_weight_3 = flatten_1.transpose(1, 2)
        flatten_1 = None
        pos_embed = torch.cat((cls_token_weight_1, pos_embed_weight_3), dim=1)
        cls_token_weight_1 = pos_embed_weight_3 = None
        add = x_2 + pos_embed
        x_2 = pos_embed = None
        x_3 = torch.nn.functional.dropout(add, 0.0, False, False)
        add = None
        key = torch.nn.functional.layer_norm(
            x_3,
            (768,),
            l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_ = (None)
        query = key.transpose(0, 1)
        key_1 = key.transpose(0, 1)
        value = key.transpose(0, 1)
        key = None
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(
            query,
            key_1,
            value,
            768,
            12,
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query = (
            key_1
        ) = (
            value
        ) = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output = multi_head_attention_forward[0]
        multi_head_attention_forward = None
        out = attn_output.transpose(0, 1)
        attn_output = None
        dropout_1 = torch.nn.functional.dropout(out, 0.0, False, False)
        out = None
        dropout_2 = torch.nn.functional.dropout(dropout_1, 0.0, False, False)
        dropout_1 = None
        output = x_3 + dropout_2
        x_3 = dropout_2 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            output,
            (768,),
            l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_ = (None)
        input_1 = torch._C._nn.linear(
            layer_norm_1,
            l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_1 = l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_2 = torch._C._nn.gelu(input_1, approximate="none")
        input_1 = None
        input_3 = torch.nn.functional.dropout(input_2, 0.0, False, False)
        input_2 = None
        input_4 = torch._C._nn.linear(
            input_3,
            l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_3 = l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_5 = torch.nn.functional.dropout(input_4, 0.0, False, False)
        input_4 = None
        output_1 = output + input_5
        output = input_5 = None
        key_2 = torch.nn.functional.layer_norm(
            output_1,
            (768,),
            l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_ = (None)
        query_1 = key_2.transpose(0, 1)
        key_3 = key_2.transpose(0, 1)
        value_1 = key_2.transpose(0, 1)
        key_2 = None
        multi_head_attention_forward_1 = torch.nn.functional.multi_head_attention_forward(
            query_1,
            key_3,
            value_1,
            768,
            12,
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_1 = (
            key_3
        ) = (
            value_1
        ) = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_1 = multi_head_attention_forward_1[0]
        multi_head_attention_forward_1 = None
        out_1 = attn_output_1.transpose(0, 1)
        attn_output_1 = None
        dropout_5 = torch.nn.functional.dropout(out_1, 0.0, False, False)
        out_1 = None
        dropout_6 = torch.nn.functional.dropout(dropout_5, 0.0, False, False)
        dropout_5 = None
        output_2 = output_1 + dropout_6
        output_1 = dropout_6 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            output_2,
            (768,),
            l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_ = (None)
        input_6 = torch._C._nn.linear(
            layer_norm_3,
            l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_3 = l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_7 = torch._C._nn.gelu(input_6, approximate="none")
        input_6 = None
        input_8 = torch.nn.functional.dropout(input_7, 0.0, False, False)
        input_7 = None
        input_9 = torch._C._nn.linear(
            input_8,
            l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_8 = l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_10 = torch.nn.functional.dropout(input_9, 0.0, False, False)
        input_9 = None
        output_3 = output_2 + input_10
        output_2 = input_10 = None
        key_4 = torch.nn.functional.layer_norm(
            output_3,
            (768,),
            l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_ = (None)
        query_2 = key_4.transpose(0, 1)
        key_5 = key_4.transpose(0, 1)
        value_2 = key_4.transpose(0, 1)
        key_4 = None
        multi_head_attention_forward_2 = torch.nn.functional.multi_head_attention_forward(
            query_2,
            key_5,
            value_2,
            768,
            12,
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_2 = (
            key_5
        ) = (
            value_2
        ) = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_2 = multi_head_attention_forward_2[0]
        multi_head_attention_forward_2 = None
        out_2 = attn_output_2.transpose(0, 1)
        attn_output_2 = None
        dropout_9 = torch.nn.functional.dropout(out_2, 0.0, False, False)
        out_2 = None
        dropout_10 = torch.nn.functional.dropout(dropout_9, 0.0, False, False)
        dropout_9 = None
        output_4 = output_3 + dropout_10
        output_3 = dropout_10 = None
        layer_norm_5 = torch.nn.functional.layer_norm(
            output_4,
            (768,),
            l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_ = (None)
        input_11 = torch._C._nn.linear(
            layer_norm_5,
            l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_5 = l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_12 = torch._C._nn.gelu(input_11, approximate="none")
        input_11 = None
        input_13 = torch.nn.functional.dropout(input_12, 0.0, False, False)
        input_12 = None
        input_14 = torch._C._nn.linear(
            input_13,
            l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_13 = l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_15 = torch.nn.functional.dropout(input_14, 0.0, False, False)
        input_14 = None
        output_5 = output_4 + input_15
        output_4 = input_15 = None
        out_3 = output_5[(slice(None, None, None), slice(1, None, None))]
        reshape_1 = out_3.reshape(1, 32, 32, 768)
        out_3 = None
        permute_1 = reshape_1.permute(0, 3, 1, 2)
        reshape_1 = None
        out_4 = permute_1.contiguous()
        permute_1 = None
        cls_token = output_5[(slice(None, None, None), 0)]
        key_6 = torch.nn.functional.layer_norm(
            output_5,
            (768,),
            l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_ = (None)
        query_3 = key_6.transpose(0, 1)
        key_7 = key_6.transpose(0, 1)
        value_3 = key_6.transpose(0, 1)
        key_6 = None
        multi_head_attention_forward_3 = torch.nn.functional.multi_head_attention_forward(
            query_3,
            key_7,
            value_3,
            768,
            12,
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_3 = (
            key_7
        ) = (
            value_3
        ) = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_3 = multi_head_attention_forward_3[0]
        multi_head_attention_forward_3 = None
        out_5 = attn_output_3.transpose(0, 1)
        attn_output_3 = None
        dropout_13 = torch.nn.functional.dropout(out_5, 0.0, False, False)
        out_5 = None
        dropout_14 = torch.nn.functional.dropout(dropout_13, 0.0, False, False)
        dropout_13 = None
        output_6 = output_5 + dropout_14
        output_5 = dropout_14 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            output_6,
            (768,),
            l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_ = (None)
        input_16 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_7 = l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_17 = torch._C._nn.gelu(input_16, approximate="none")
        input_16 = None
        input_18 = torch.nn.functional.dropout(input_17, 0.0, False, False)
        input_17 = None
        input_19 = torch._C._nn.linear(
            input_18,
            l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_18 = l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_20 = torch.nn.functional.dropout(input_19, 0.0, False, False)
        input_19 = None
        output_7 = output_6 + input_20
        output_6 = input_20 = None
        key_8 = torch.nn.functional.layer_norm(
            output_7,
            (768,),
            l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_ = (None)
        query_4 = key_8.transpose(0, 1)
        key_9 = key_8.transpose(0, 1)
        value_4 = key_8.transpose(0, 1)
        key_8 = None
        multi_head_attention_forward_4 = torch.nn.functional.multi_head_attention_forward(
            query_4,
            key_9,
            value_4,
            768,
            12,
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_4 = (
            key_9
        ) = (
            value_4
        ) = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_4 = multi_head_attention_forward_4[0]
        multi_head_attention_forward_4 = None
        out_6 = attn_output_4.transpose(0, 1)
        attn_output_4 = None
        dropout_17 = torch.nn.functional.dropout(out_6, 0.0, False, False)
        out_6 = None
        dropout_18 = torch.nn.functional.dropout(dropout_17, 0.0, False, False)
        dropout_17 = None
        output_8 = output_7 + dropout_18
        output_7 = dropout_18 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            output_8,
            (768,),
            l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_ = (None)
        input_21 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_9 = l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_22 = torch._C._nn.gelu(input_21, approximate="none")
        input_21 = None
        input_23 = torch.nn.functional.dropout(input_22, 0.0, False, False)
        input_22 = None
        input_24 = torch._C._nn.linear(
            input_23,
            l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_23 = l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_25 = torch.nn.functional.dropout(input_24, 0.0, False, False)
        input_24 = None
        output_9 = output_8 + input_25
        output_8 = input_25 = None
        key_10 = torch.nn.functional.layer_norm(
            output_9,
            (768,),
            l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_ = (None)
        query_5 = key_10.transpose(0, 1)
        key_11 = key_10.transpose(0, 1)
        value_5 = key_10.transpose(0, 1)
        key_10 = None
        multi_head_attention_forward_5 = torch.nn.functional.multi_head_attention_forward(
            query_5,
            key_11,
            value_5,
            768,
            12,
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_5 = (
            key_11
        ) = (
            value_5
        ) = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_5 = multi_head_attention_forward_5[0]
        multi_head_attention_forward_5 = None
        out_7 = attn_output_5.transpose(0, 1)
        attn_output_5 = None
        dropout_21 = torch.nn.functional.dropout(out_7, 0.0, False, False)
        out_7 = None
        dropout_22 = torch.nn.functional.dropout(dropout_21, 0.0, False, False)
        dropout_21 = None
        output_10 = output_9 + dropout_22
        output_9 = dropout_22 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            output_10,
            (768,),
            l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_ = (None)
        input_26 = torch._C._nn.linear(
            layer_norm_11,
            l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_11 = l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_27 = torch._C._nn.gelu(input_26, approximate="none")
        input_26 = None
        input_28 = torch.nn.functional.dropout(input_27, 0.0, False, False)
        input_27 = None
        input_29 = torch._C._nn.linear(
            input_28,
            l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_28 = l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_30 = torch.nn.functional.dropout(input_29, 0.0, False, False)
        input_29 = None
        output_11 = output_10 + input_30
        output_10 = input_30 = None
        out_8 = output_11[(slice(None, None, None), slice(1, None, None))]
        reshape_2 = out_8.reshape(1, 32, 32, 768)
        out_8 = None
        permute_2 = reshape_2.permute(0, 3, 1, 2)
        reshape_2 = None
        out_9 = permute_2.contiguous()
        permute_2 = None
        cls_token_1 = output_11[(slice(None, None, None), 0)]
        key_12 = torch.nn.functional.layer_norm(
            output_11,
            (768,),
            l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_ = (None)
        query_6 = key_12.transpose(0, 1)
        key_13 = key_12.transpose(0, 1)
        value_6 = key_12.transpose(0, 1)
        key_12 = None
        multi_head_attention_forward_6 = torch.nn.functional.multi_head_attention_forward(
            query_6,
            key_13,
            value_6,
            768,
            12,
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_6 = (
            key_13
        ) = (
            value_6
        ) = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_6 = multi_head_attention_forward_6[0]
        multi_head_attention_forward_6 = None
        out_10 = attn_output_6.transpose(0, 1)
        attn_output_6 = None
        dropout_25 = torch.nn.functional.dropout(out_10, 0.0, False, False)
        out_10 = None
        dropout_26 = torch.nn.functional.dropout(dropout_25, 0.0, False, False)
        dropout_25 = None
        output_12 = output_11 + dropout_26
        output_11 = dropout_26 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            output_12,
            (768,),
            l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_ = (None)
        input_31 = torch._C._nn.linear(
            layer_norm_13,
            l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_13 = l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_32 = torch._C._nn.gelu(input_31, approximate="none")
        input_31 = None
        input_33 = torch.nn.functional.dropout(input_32, 0.0, False, False)
        input_32 = None
        input_34 = torch._C._nn.linear(
            input_33,
            l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_33 = l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_35 = torch.nn.functional.dropout(input_34, 0.0, False, False)
        input_34 = None
        output_13 = output_12 + input_35
        output_12 = input_35 = None
        key_14 = torch.nn.functional.layer_norm(
            output_13,
            (768,),
            l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_ = (None)
        query_7 = key_14.transpose(0, 1)
        key_15 = key_14.transpose(0, 1)
        value_7 = key_14.transpose(0, 1)
        key_14 = None
        multi_head_attention_forward_7 = torch.nn.functional.multi_head_attention_forward(
            query_7,
            key_15,
            value_7,
            768,
            12,
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_7 = (
            key_15
        ) = (
            value_7
        ) = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_7 = multi_head_attention_forward_7[0]
        multi_head_attention_forward_7 = None
        out_11 = attn_output_7.transpose(0, 1)
        attn_output_7 = None
        dropout_29 = torch.nn.functional.dropout(out_11, 0.0, False, False)
        out_11 = None
        dropout_30 = torch.nn.functional.dropout(dropout_29, 0.0, False, False)
        dropout_29 = None
        output_14 = output_13 + dropout_30
        output_13 = dropout_30 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            output_14,
            (768,),
            l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_ = (None)
        input_36 = torch._C._nn.linear(
            layer_norm_15,
            l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_15 = l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_37 = torch._C._nn.gelu(input_36, approximate="none")
        input_36 = None
        input_38 = torch.nn.functional.dropout(input_37, 0.0, False, False)
        input_37 = None
        input_39 = torch._C._nn.linear(
            input_38,
            l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_38 = l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_40 = torch.nn.functional.dropout(input_39, 0.0, False, False)
        input_39 = None
        output_15 = output_14 + input_40
        output_14 = input_40 = None
        key_16 = torch.nn.functional.layer_norm(
            output_15,
            (768,),
            l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_ = (None)
        query_8 = key_16.transpose(0, 1)
        key_17 = key_16.transpose(0, 1)
        value_8 = key_16.transpose(0, 1)
        key_16 = None
        multi_head_attention_forward_8 = torch.nn.functional.multi_head_attention_forward(
            query_8,
            key_17,
            value_8,
            768,
            12,
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_8 = (
            key_17
        ) = (
            value_8
        ) = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_8 = multi_head_attention_forward_8[0]
        multi_head_attention_forward_8 = None
        out_12 = attn_output_8.transpose(0, 1)
        attn_output_8 = None
        dropout_33 = torch.nn.functional.dropout(out_12, 0.0, False, False)
        out_12 = None
        dropout_34 = torch.nn.functional.dropout(dropout_33, 0.0, False, False)
        dropout_33 = None
        output_16 = output_15 + dropout_34
        output_15 = dropout_34 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            output_16,
            (768,),
            l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_ = (None)
        input_41 = torch._C._nn.linear(
            layer_norm_17,
            l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_17 = l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_42 = torch._C._nn.gelu(input_41, approximate="none")
        input_41 = None
        input_43 = torch.nn.functional.dropout(input_42, 0.0, False, False)
        input_42 = None
        input_44 = torch._C._nn.linear(
            input_43,
            l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_43 = l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_45 = torch.nn.functional.dropout(input_44, 0.0, False, False)
        input_44 = None
        output_17 = output_16 + input_45
        output_16 = input_45 = None
        out_13 = output_17[(slice(None, None, None), slice(1, None, None))]
        reshape_3 = out_13.reshape(1, 32, 32, 768)
        out_13 = None
        permute_3 = reshape_3.permute(0, 3, 1, 2)
        reshape_3 = None
        out_14 = permute_3.contiguous()
        permute_3 = None
        cls_token_2 = output_17[(slice(None, None, None), 0)]
        key_18 = torch.nn.functional.layer_norm(
            output_17,
            (768,),
            l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_ = (None)
        query_9 = key_18.transpose(0, 1)
        key_19 = key_18.transpose(0, 1)
        value_9 = key_18.transpose(0, 1)
        key_18 = None
        multi_head_attention_forward_9 = torch.nn.functional.multi_head_attention_forward(
            query_9,
            key_19,
            value_9,
            768,
            12,
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_9 = (
            key_19
        ) = (
            value_9
        ) = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_9 = multi_head_attention_forward_9[0]
        multi_head_attention_forward_9 = None
        out_15 = attn_output_9.transpose(0, 1)
        attn_output_9 = None
        dropout_37 = torch.nn.functional.dropout(out_15, 0.0, False, False)
        out_15 = None
        dropout_38 = torch.nn.functional.dropout(dropout_37, 0.0, False, False)
        dropout_37 = None
        output_18 = output_17 + dropout_38
        output_17 = dropout_38 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            output_18,
            (768,),
            l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_ = (None)
        input_46 = torch._C._nn.linear(
            layer_norm_19,
            l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_19 = l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_47 = torch._C._nn.gelu(input_46, approximate="none")
        input_46 = None
        input_48 = torch.nn.functional.dropout(input_47, 0.0, False, False)
        input_47 = None
        input_49 = torch._C._nn.linear(
            input_48,
            l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_48 = l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_50 = torch.nn.functional.dropout(input_49, 0.0, False, False)
        input_49 = None
        output_19 = output_18 + input_50
        output_18 = input_50 = None
        key_20 = torch.nn.functional.layer_norm(
            output_19,
            (768,),
            l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_ = (None)
        query_10 = key_20.transpose(0, 1)
        key_21 = key_20.transpose(0, 1)
        value_10 = key_20.transpose(0, 1)
        key_20 = None
        multi_head_attention_forward_10 = torch.nn.functional.multi_head_attention_forward(
            query_10,
            key_21,
            value_10,
            768,
            12,
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_10 = (
            key_21
        ) = (
            value_10
        ) = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_10 = multi_head_attention_forward_10[0]
        multi_head_attention_forward_10 = None
        out_16 = attn_output_10.transpose(0, 1)
        attn_output_10 = None
        dropout_41 = torch.nn.functional.dropout(out_16, 0.0, False, False)
        out_16 = None
        dropout_42 = torch.nn.functional.dropout(dropout_41, 0.0, False, False)
        dropout_41 = None
        output_20 = output_19 + dropout_42
        output_19 = dropout_42 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            output_20,
            (768,),
            l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_ = (None)
        input_51 = torch._C._nn.linear(
            layer_norm_21,
            l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_21 = l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_52 = torch._C._nn.gelu(input_51, approximate="none")
        input_51 = None
        input_53 = torch.nn.functional.dropout(input_52, 0.0, False, False)
        input_52 = None
        input_54 = torch._C._nn.linear(
            input_53,
            l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_53 = l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_55 = torch.nn.functional.dropout(input_54, 0.0, False, False)
        input_54 = None
        output_21 = output_20 + input_55
        output_20 = input_55 = None
        key_22 = torch.nn.functional.layer_norm(
            output_21,
            (768,),
            l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_ = (None)
        query_11 = key_22.transpose(0, 1)
        key_23 = key_22.transpose(0, 1)
        value_11 = key_22.transpose(0, 1)
        key_22 = None
        multi_head_attention_forward_11 = torch.nn.functional.multi_head_attention_forward(
            query_11,
            key_23,
            value_11,
            768,
            12,
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_11 = (
            key_23
        ) = (
            value_11
        ) = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_11 = multi_head_attention_forward_11[0]
        multi_head_attention_forward_11 = None
        out_17 = attn_output_11.transpose(0, 1)
        attn_output_11 = None
        dropout_45 = torch.nn.functional.dropout(out_17, 0.0, False, False)
        out_17 = None
        dropout_46 = torch.nn.functional.dropout(dropout_45, 0.0, False, False)
        dropout_45 = None
        output_22 = output_21 + dropout_46
        output_21 = dropout_46 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            output_22,
            (768,),
            l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_ = (None)
        input_56 = torch._C._nn.linear(
            layer_norm_23,
            l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        layer_norm_23 = l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_57 = torch._C._nn.gelu(input_56, approximate="none")
        input_56 = None
        input_58 = torch.nn.functional.dropout(input_57, 0.0, False, False)
        input_57 = None
        input_59 = torch._C._nn.linear(
            input_58,
            l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_58 = l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_60 = torch.nn.functional.dropout(input_59, 0.0, False, False)
        input_59 = None
        output_23 = output_22 + input_60
        output_22 = input_60 = None
        out_18 = output_23[(slice(None, None, None), slice(1, None, None))]
        reshape_4 = out_18.reshape(1, 32, 32, 768)
        out_18 = None
        permute_4 = reshape_4.permute(0, 3, 1, 2)
        reshape_4 = None
        out_19 = permute_4.contiguous()
        permute_4 = None
        cls_token_3 = output_23[(slice(None, None, None), 0)]
        output_23 = None
        flatten_2 = out_4.flatten(2)
        out_4 = None
        x_4 = flatten_2.permute((0, 2, 1))
        flatten_2 = None
        unsqueeze_1 = cls_token.unsqueeze(1)
        cls_token = None
        readout = unsqueeze_1.expand_as(x_4)
        unsqueeze_1 = None
        cat_2 = torch.cat((x_4, readout), -1)
        x_4 = readout = None
        input_61 = torch._C._nn.linear(
            cat_2,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_0_modules_0_parameters_weight_,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_0_modules_0_parameters_bias_,
        )
        cat_2 = l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_0_modules_0_parameters_weight_ = l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_0_modules_0_parameters_bias_ = (None)
        input_62 = torch._C._nn.gelu(input_61, approximate="none")
        input_61 = None
        permute_6 = input_62.permute(0, 2, 1)
        input_62 = None
        x_5 = permute_6.reshape((1, 768, 32, 32))
        permute_6 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_0_modules_conv_parameters_weight_,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_0_modules_conv_parameters_weight_ = l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_0_modules_conv_parameters_bias_ = (None)
        x_7 = torch.conv_transpose2d(
            x_6,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_0_parameters_weight_,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_0_parameters_bias_,
            (4, 4),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        x_6 = l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_0_parameters_weight_ = l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_0_parameters_bias_ = (None)
        flatten_3 = out_9.flatten(2)
        out_9 = None
        x_8 = flatten_3.permute((0, 2, 1))
        flatten_3 = None
        unsqueeze_2 = cls_token_1.unsqueeze(1)
        cls_token_1 = None
        readout_1 = unsqueeze_2.expand_as(x_8)
        unsqueeze_2 = None
        cat_3 = torch.cat((x_8, readout_1), -1)
        x_8 = readout_1 = None
        input_63 = torch._C._nn.linear(
            cat_3,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_1_modules_0_parameters_weight_,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_1_modules_0_parameters_bias_,
        )
        cat_3 = l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_1_modules_0_parameters_weight_ = l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_1_modules_0_parameters_bias_ = (None)
        input_64 = torch._C._nn.gelu(input_63, approximate="none")
        input_63 = None
        permute_8 = input_64.permute(0, 2, 1)
        input_64 = None
        x_9 = permute_8.reshape((1, 768, 32, 32))
        permute_8 = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_1_modules_conv_parameters_weight_,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_1_modules_conv_parameters_weight_ = l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_1_modules_conv_parameters_bias_ = (None)
        x_11 = torch.conv_transpose2d(
            x_10,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_1_parameters_weight_,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        x_10 = l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_1_parameters_weight_ = l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_1_parameters_bias_ = (None)
        flatten_4 = out_14.flatten(2)
        out_14 = None
        x_12 = flatten_4.permute((0, 2, 1))
        flatten_4 = None
        unsqueeze_3 = cls_token_2.unsqueeze(1)
        cls_token_2 = None
        readout_2 = unsqueeze_3.expand_as(x_12)
        unsqueeze_3 = None
        cat_4 = torch.cat((x_12, readout_2), -1)
        x_12 = readout_2 = None
        input_65 = torch._C._nn.linear(
            cat_4,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_2_modules_0_parameters_weight_,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_2_modules_0_parameters_bias_,
        )
        cat_4 = l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_2_modules_0_parameters_weight_ = l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_2_modules_0_parameters_bias_ = (None)
        input_66 = torch._C._nn.gelu(input_65, approximate="none")
        input_65 = None
        permute_10 = input_66.permute(0, 2, 1)
        input_66 = None
        x_13 = permute_10.reshape((1, 768, 32, 32))
        permute_10 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_2_modules_conv_parameters_weight_,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_2_modules_conv_parameters_weight_ = l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_2_modules_conv_parameters_bias_ = (None)
        flatten_5 = out_19.flatten(2)
        out_19 = None
        x_15 = flatten_5.permute((0, 2, 1))
        flatten_5 = None
        unsqueeze_4 = cls_token_3.unsqueeze(1)
        cls_token_3 = None
        readout_3 = unsqueeze_4.expand_as(x_15)
        unsqueeze_4 = None
        cat_5 = torch.cat((x_15, readout_3), -1)
        x_15 = readout_3 = None
        input_67 = torch._C._nn.linear(
            cat_5,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_3_modules_0_parameters_weight_,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_3_modules_0_parameters_bias_,
        )
        cat_5 = l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_3_modules_0_parameters_weight_ = l_self_modules_decode_head_modules_reassemble_blocks_modules_readout_projects_modules_3_modules_0_parameters_bias_ = (None)
        input_68 = torch._C._nn.gelu(input_67, approximate="none")
        input_67 = None
        permute_12 = input_68.permute(0, 2, 1)
        input_68 = None
        x_16 = permute_12.reshape((1, 768, 32, 32))
        permute_12 = None
        x_17 = torch.conv2d(
            x_16,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_3_modules_conv_parameters_weight_,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_3_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_3_modules_conv_parameters_weight_ = l_self_modules_decode_head_modules_reassemble_blocks_modules_projects_modules_3_modules_conv_parameters_bias_ = (None)
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_3_parameters_weight_,
            l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_3_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_3_parameters_weight_ = l_self_modules_decode_head_modules_reassemble_blocks_modules_resize_layers_modules_3_parameters_bias_ = (None)
        x_19 = torch.conv2d(
            x_7,
            l_self_modules_decode_head_modules_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_decode_head_modules_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_20 = torch.conv2d(
            x_11,
            l_self_modules_decode_head_modules_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_decode_head_modules_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_21 = torch.conv2d(
            x_14,
            l_self_modules_decode_head_modules_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_decode_head_modules_convs_modules_2_modules_conv_parameters_weight_ = (None)
        x_22 = torch.conv2d(
            x_18,
            l_self_modules_decode_head_modules_convs_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_decode_head_modules_convs_modules_3_modules_conv_parameters_weight_ = (None)
        inputs_ = x_22.clone()
        x_23 = torch.nn.functional.relu(x_22, inplace=True)
        x_22 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_26 = torch.nn.functional.relu(x_25, inplace=True)
        x_25 = None
        x_27 = torch.conv2d(
            x_26,
            l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_27 = l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_29 = x_28 + inputs_
        x_28 = inputs_ = None
        x_30 = torch.nn.functional.interpolate(x_29, None, 2, "bilinear", True)
        x_29 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_project_modules_conv_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_project_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_project_modules_conv_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_0_modules_project_modules_conv_parameters_bias_ = (None)
        inputs__1 = x_21.clone()
        x_32 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_35 = torch.nn.functional.relu(x_34, inplace=True)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_26 = x_37 + inputs__1
        x_37 = inputs__1 = None
        x_38 = x_31 + add_26
        x_31 = add_26 = None
        inputs__2 = x_38.clone()
        x_39 = torch.nn.functional.relu(x_38, inplace=True)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_42 = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_45 = x_44 + inputs__2
        x_44 = inputs__2 = None
        x_46 = torch.nn.functional.interpolate(x_45, None, 2, "bilinear", True)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_project_modules_conv_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_project_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_project_modules_conv_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_1_modules_project_modules_conv_parameters_bias_ = (None)
        inputs__3 = x_20.clone()
        x_48 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_51 = torch.nn.functional.relu(x_50, inplace=True)
        x_50 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_51 = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_29 = x_53 + inputs__3
        x_53 = inputs__3 = None
        x_54 = x_47 + add_29
        x_47 = add_29 = None
        inputs__4 = x_54.clone()
        x_55 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_58 = torch.nn.functional.relu(x_57, inplace=True)
        x_57 = None
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_58 = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_61 = x_60 + inputs__4
        x_60 = inputs__4 = None
        x_62 = torch.nn.functional.interpolate(x_61, None, 2, "bilinear", True)
        x_61 = None
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_project_modules_conv_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_project_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_62 = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_project_modules_conv_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_2_modules_project_modules_conv_parameters_bias_ = (None)
        inputs__5 = x_19.clone()
        x_64 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_32 = x_69 + inputs__5
        x_69 = inputs__5 = None
        x_70 = x_63 + add_32
        x_63 = add_32 = None
        inputs__6 = x_70.clone()
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_71 = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_res_conv_unit2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_77 = x_76 + inputs__6
        x_76 = inputs__6 = None
        x_78 = torch.nn.functional.interpolate(x_77, None, 2, "bilinear", True)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_project_modules_conv_parameters_weight_,
            l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_project_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_project_modules_conv_parameters_weight_ = l_self_modules_decode_head_modules_fusion_blocks_modules_3_modules_project_modules_conv_parameters_bias_ = (None)
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_decode_head_modules_project_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_79 = (
            l_self_modules_decode_head_modules_project_modules_conv_parameters_weight_
        ) = None
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_decode_head_modules_project_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_project_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_project_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_project_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = (
            l_self_modules_decode_head_modules_project_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_decode_head_modules_project_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_decode_head_modules_project_modules_bn_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_project_modules_bn_parameters_bias_
        ) = None
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        feat = torch.nn.functional.dropout2d(x_82, 0.1, False, False)
        x_82 = None
        output_24 = torch.conv2d(
            feat,
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_,
            l_self_modules_decode_head_modules_conv_seg_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        feat = (
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_
        ) = l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = None
        return (output_24,)
