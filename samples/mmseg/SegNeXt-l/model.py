import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block1_modules_0_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block1_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block1_modules_0_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block1_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block1_modules_1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block1_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block1_modules_1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block1_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block1_modules_2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block1_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block1_modules_2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block1_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed2_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed2_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_0_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_0_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_3_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_3_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_4_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_4_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block2_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed3_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed3_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_0_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_0_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_3_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_3_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_4_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_4_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_5_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_5_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_6_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_6_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_7_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_7_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_8_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_8_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_9_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_9_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_10_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_10_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_11_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_11_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_12_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_12_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_13_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_13_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_14_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_14_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_15_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_15_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_16_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_16_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_17_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_17_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_18_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_18_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_19_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_19_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_20_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_20_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_21_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_21_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_22_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_22_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_23_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_23_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_24_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_24_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_24_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_24_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_25_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_25_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_25_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_25_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_26_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_26_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_26_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block3_modules_26_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed4_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed4_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed4_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed4_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_patch_embed4_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed4_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block4_modules_0_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block4_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block4_modules_0_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block4_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block4_modules_1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block4_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block4_modules_1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block4_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block4_modules_2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block4_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_block4_modules_2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_block4_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_squeeze_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_squeeze_modules_gn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_squeeze_modules_gn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_hamburger_modules_ham_in_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_hamburger_modules_ham_in_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_gn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_gn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_align_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_align_modules_gn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_align_modules_gn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_0_parameters_weight_ = L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_0_parameters_weight_
        l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_0_parameters_bias_ = L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_0_parameters_bias_
        l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_parameters_weight_ = L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_parameters_weight_
        l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_parameters_bias_ = L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_parameters_bias_
        l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_3_parameters_weight_ = L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_3_parameters_weight_
        l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_3_parameters_bias_ = L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_3_parameters_bias_
        l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_buffers_running_mean_ = L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_buffers_running_mean_
        l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_buffers_running_var_ = L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_buffers_running_var_
        l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_parameters_weight_ = L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_parameters_weight_
        l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_parameters_bias_ = L_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block1_modules_0_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block1_modules_0_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block1_modules_0_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block1_modules_0_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block1_modules_0_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block1_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block1_modules_0_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block1_modules_0_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block1_modules_0_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block1_modules_0_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block1_modules_0_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block1_modules_0_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block1_modules_1_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block1_modules_1_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block1_modules_1_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block1_modules_1_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block1_modules_1_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block1_modules_1_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block1_modules_1_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block1_modules_1_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block1_modules_1_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block1_modules_1_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block1_modules_1_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block1_modules_1_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block1_modules_2_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block1_modules_2_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block1_modules_2_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block1_modules_2_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block1_modules_2_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block1_modules_2_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block1_modules_2_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block1_modules_2_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block1_modules_2_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block1_modules_2_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block1_modules_2_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block1_modules_2_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_norm1_parameters_weight_ = (
            L_self_modules_backbone_modules_norm1_parameters_weight_
        )
        l_self_modules_backbone_modules_norm1_parameters_bias_ = (
            L_self_modules_backbone_modules_norm1_parameters_bias_
        )
        l_self_modules_backbone_modules_patch_embed2_modules_proj_parameters_weight_ = (
            L_self_modules_backbone_modules_patch_embed2_modules_proj_parameters_weight_
        )
        l_self_modules_backbone_modules_patch_embed2_modules_proj_parameters_bias_ = (
            L_self_modules_backbone_modules_patch_embed2_modules_proj_parameters_bias_
        )
        l_self_modules_backbone_modules_patch_embed2_modules_norm_buffers_running_mean_ = L_self_modules_backbone_modules_patch_embed2_modules_norm_buffers_running_mean_
        l_self_modules_backbone_modules_patch_embed2_modules_norm_buffers_running_var_ = L_self_modules_backbone_modules_patch_embed2_modules_norm_buffers_running_var_
        l_self_modules_backbone_modules_patch_embed2_modules_norm_parameters_weight_ = (
            L_self_modules_backbone_modules_patch_embed2_modules_norm_parameters_weight_
        )
        l_self_modules_backbone_modules_patch_embed2_modules_norm_parameters_bias_ = (
            L_self_modules_backbone_modules_patch_embed2_modules_norm_parameters_bias_
        )
        l_self_modules_backbone_modules_block2_modules_0_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block2_modules_0_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block2_modules_0_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block2_modules_0_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block2_modules_0_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block2_modules_0_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block2_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_0_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block2_modules_0_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block2_modules_0_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block2_modules_0_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block2_modules_0_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block2_modules_0_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block2_modules_0_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block2_modules_1_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block2_modules_1_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block2_modules_1_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block2_modules_1_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block2_modules_1_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block2_modules_1_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block2_modules_1_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block2_modules_1_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block2_modules_1_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block2_modules_1_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block2_modules_1_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block2_modules_1_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block2_modules_2_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block2_modules_2_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block2_modules_2_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block2_modules_2_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block2_modules_2_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block2_modules_2_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block2_modules_2_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block2_modules_2_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block2_modules_2_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block2_modules_2_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block2_modules_2_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block2_modules_2_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block2_modules_3_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block2_modules_3_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block2_modules_3_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block2_modules_3_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block2_modules_3_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block2_modules_3_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block2_modules_3_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block2_modules_3_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block2_modules_3_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block2_modules_3_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block2_modules_3_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block2_modules_3_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block2_modules_4_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block2_modules_4_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block2_modules_4_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block2_modules_4_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block2_modules_4_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block2_modules_4_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block2_modules_4_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block2_modules_4_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block2_modules_4_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block2_modules_4_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block2_modules_4_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block2_modules_4_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_norm2_parameters_weight_ = (
            L_self_modules_backbone_modules_norm2_parameters_weight_
        )
        l_self_modules_backbone_modules_norm2_parameters_bias_ = (
            L_self_modules_backbone_modules_norm2_parameters_bias_
        )
        l_self_modules_backbone_modules_patch_embed3_modules_proj_parameters_weight_ = (
            L_self_modules_backbone_modules_patch_embed3_modules_proj_parameters_weight_
        )
        l_self_modules_backbone_modules_patch_embed3_modules_proj_parameters_bias_ = (
            L_self_modules_backbone_modules_patch_embed3_modules_proj_parameters_bias_
        )
        l_self_modules_backbone_modules_patch_embed3_modules_norm_buffers_running_mean_ = L_self_modules_backbone_modules_patch_embed3_modules_norm_buffers_running_mean_
        l_self_modules_backbone_modules_patch_embed3_modules_norm_buffers_running_var_ = L_self_modules_backbone_modules_patch_embed3_modules_norm_buffers_running_var_
        l_self_modules_backbone_modules_patch_embed3_modules_norm_parameters_weight_ = (
            L_self_modules_backbone_modules_patch_embed3_modules_norm_parameters_weight_
        )
        l_self_modules_backbone_modules_patch_embed3_modules_norm_parameters_bias_ = (
            L_self_modules_backbone_modules_patch_embed3_modules_norm_parameters_bias_
        )
        l_self_modules_backbone_modules_block3_modules_0_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_0_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_0_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_0_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_0_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_0_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_0_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_0_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_0_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_0_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_0_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_0_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_0_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_1_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_1_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_1_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_1_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_1_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_1_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_1_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_1_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_1_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_1_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_1_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_1_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_2_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_2_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_2_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_2_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_2_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_2_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_2_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_2_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_2_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_2_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_2_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_2_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_3_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_3_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_3_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_3_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_3_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_3_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_3_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_3_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_3_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_3_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_3_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_3_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_4_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_4_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_4_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_4_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_4_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_4_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_4_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_4_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_4_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_4_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_4_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_4_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_5_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_5_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_5_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_5_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_5_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_5_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_5_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_5_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_5_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_5_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_5_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_5_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_6_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_6_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_6_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_6_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_6_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_6_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_6_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_6_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_6_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_6_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_6_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_6_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_7_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_7_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_7_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_7_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_7_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_7_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_7_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_7_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_7_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_7_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_7_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_7_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_8_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_8_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_8_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_8_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_8_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_8_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_8_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_8_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_8_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_8_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_8_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_8_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_9_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_9_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_9_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_9_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_9_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_9_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_9_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_9_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_9_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_9_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_9_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_9_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_10_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_10_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_10_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_10_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_10_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_10_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_10_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_10_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_10_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_10_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_10_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_10_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_11_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_11_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_11_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_11_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_11_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_11_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_11_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_11_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_11_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_11_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_11_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_11_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_12_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_12_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_12_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_12_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_12_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_12_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_12_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_12_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_12_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_12_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_12_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_12_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_13_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_13_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_13_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_13_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_13_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_13_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_13_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_13_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_13_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_13_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_13_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_13_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_14_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_14_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_14_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_14_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_14_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_14_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_14_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_14_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_14_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_14_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_14_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_14_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_15_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_15_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_15_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_15_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_15_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_15_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_15_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_15_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_15_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_15_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_15_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_15_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_16_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_16_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_16_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_16_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_16_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_16_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_16_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_16_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_16_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_16_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_16_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_16_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_17_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_17_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_17_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_17_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_17_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_17_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_17_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_17_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_17_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_17_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_17_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_17_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_18_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_18_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_18_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_18_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_18_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_18_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_18_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_18_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_18_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_18_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_18_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_18_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_19_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_19_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_19_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_19_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_19_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_19_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_19_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_19_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_19_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_19_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_19_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_19_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_20_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_20_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_20_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_20_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_20_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_20_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_20_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_20_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_20_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_20_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_20_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_20_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_21_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_21_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_21_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_21_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_21_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_21_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_21_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_21_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_21_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_21_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_21_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_21_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_22_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_22_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_22_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_22_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_22_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_22_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_22_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_22_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_22_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_22_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_22_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_22_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_23_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_23_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_23_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_23_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_23_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_23_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_23_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_23_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_23_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_23_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_23_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_23_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_24_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_24_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_24_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_24_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_24_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_24_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_24_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_24_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_24_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_24_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_24_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_24_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_25_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_25_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_25_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_25_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_25_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_25_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_25_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_25_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_25_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_25_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_25_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_25_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block3_modules_26_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block3_modules_26_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_26_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_26_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_26_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_26_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block3_modules_26_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block3_modules_26_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block3_modules_26_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block3_modules_26_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block3_modules_26_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block3_modules_26_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_norm3_parameters_weight_ = (
            L_self_modules_backbone_modules_norm3_parameters_weight_
        )
        l_self_modules_backbone_modules_norm3_parameters_bias_ = (
            L_self_modules_backbone_modules_norm3_parameters_bias_
        )
        l_self_modules_backbone_modules_patch_embed4_modules_proj_parameters_weight_ = (
            L_self_modules_backbone_modules_patch_embed4_modules_proj_parameters_weight_
        )
        l_self_modules_backbone_modules_patch_embed4_modules_proj_parameters_bias_ = (
            L_self_modules_backbone_modules_patch_embed4_modules_proj_parameters_bias_
        )
        l_self_modules_backbone_modules_patch_embed4_modules_norm_buffers_running_mean_ = L_self_modules_backbone_modules_patch_embed4_modules_norm_buffers_running_mean_
        l_self_modules_backbone_modules_patch_embed4_modules_norm_buffers_running_var_ = L_self_modules_backbone_modules_patch_embed4_modules_norm_buffers_running_var_
        l_self_modules_backbone_modules_patch_embed4_modules_norm_parameters_weight_ = (
            L_self_modules_backbone_modules_patch_embed4_modules_norm_parameters_weight_
        )
        l_self_modules_backbone_modules_patch_embed4_modules_norm_parameters_bias_ = (
            L_self_modules_backbone_modules_patch_embed4_modules_norm_parameters_bias_
        )
        l_self_modules_backbone_modules_block4_modules_0_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block4_modules_0_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block4_modules_0_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block4_modules_0_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block4_modules_0_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block4_modules_0_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block4_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_0_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block4_modules_0_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block4_modules_0_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block4_modules_0_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block4_modules_0_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block4_modules_0_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block4_modules_0_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block4_modules_1_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block4_modules_1_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block4_modules_1_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block4_modules_1_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block4_modules_1_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block4_modules_1_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block4_modules_1_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block4_modules_1_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block4_modules_1_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block4_modules_1_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block4_modules_1_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block4_modules_1_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_parameters_layer_scale_1_ = (
            L_self_modules_backbone_modules_block4_modules_2_parameters_layer_scale_1_
        )
        l_self_modules_backbone_modules_block4_modules_2_modules_norm1_buffers_running_mean_ = L_self_modules_backbone_modules_block4_modules_2_modules_norm1_buffers_running_mean_
        l_self_modules_backbone_modules_block4_modules_2_modules_norm1_buffers_running_var_ = L_self_modules_backbone_modules_block4_modules_2_modules_norm1_buffers_running_var_
        l_self_modules_backbone_modules_block4_modules_2_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_parameters_layer_scale_2_ = (
            L_self_modules_backbone_modules_block4_modules_2_parameters_layer_scale_2_
        )
        l_self_modules_backbone_modules_block4_modules_2_modules_norm2_buffers_running_mean_ = L_self_modules_backbone_modules_block4_modules_2_modules_norm2_buffers_running_mean_
        l_self_modules_backbone_modules_block4_modules_2_modules_norm2_buffers_running_var_ = L_self_modules_backbone_modules_block4_modules_2_modules_norm2_buffers_running_var_
        l_self_modules_backbone_modules_block4_modules_2_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_dwconv_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_dwconv_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_dwconv_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_dwconv_parameters_bias_
        l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_norm4_parameters_weight_ = (
            L_self_modules_backbone_modules_norm4_parameters_weight_
        )
        l_self_modules_backbone_modules_norm4_parameters_bias_ = (
            L_self_modules_backbone_modules_norm4_parameters_bias_
        )
        l_self_modules_decode_head_modules_squeeze_modules_conv_parameters_weight_ = (
            L_self_modules_decode_head_modules_squeeze_modules_conv_parameters_weight_
        )
        l_self_modules_decode_head_modules_squeeze_modules_gn_parameters_weight_ = (
            L_self_modules_decode_head_modules_squeeze_modules_gn_parameters_weight_
        )
        l_self_modules_decode_head_modules_squeeze_modules_gn_parameters_bias_ = (
            L_self_modules_decode_head_modules_squeeze_modules_gn_parameters_bias_
        )
        l_self_modules_decode_head_modules_hamburger_modules_ham_in_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_hamburger_modules_ham_in_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_hamburger_modules_ham_in_modules_conv_parameters_bias_ = L_self_modules_decode_head_modules_hamburger_modules_ham_in_modules_conv_parameters_bias_
        l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_gn_parameters_weight_ = L_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_gn_parameters_weight_
        l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_gn_parameters_bias_ = L_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_gn_parameters_bias_
        l_self_modules_decode_head_modules_align_modules_conv_parameters_weight_ = (
            L_self_modules_decode_head_modules_align_modules_conv_parameters_weight_
        )
        l_self_modules_decode_head_modules_align_modules_gn_parameters_weight_ = (
            L_self_modules_decode_head_modules_align_modules_gn_parameters_weight_
        )
        l_self_modules_decode_head_modules_align_modules_gn_parameters_bias_ = (
            L_self_modules_decode_head_modules_align_modules_gn_parameters_bias_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_weight_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_bias_
        )
        input_1 = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_0_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_0_parameters_weight_ = l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_0_parameters_bias_ = (None)
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_parameters_weight_ = l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_1_parameters_bias_ = (None)
        input_3 = torch._C._nn.gelu(input_2, approximate="none")
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_3_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_3_parameters_weight_ = l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_3_parameters_bias_ = (None)
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_buffers_running_mean_,
            l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_buffers_running_var_,
            l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_buffers_running_mean_ = l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_buffers_running_var_ = l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_parameters_weight_ = l_self_modules_backbone_modules_patch_embed1_modules_proj_modules_4_parameters_bias_ = (None)
        flatten = input_5.flatten(2)
        input_5 = None
        x = flatten.transpose(1, 2)
        flatten = None
        permute = x.permute(0, 2, 1)
        x = None
        x_1 = permute.view(1, 64, 128, 128)
        permute = None
        unsqueeze = l_self_modules_backbone_modules_block1_modules_0_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block1_modules_0_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_1 = unsqueeze.unsqueeze(-1)
        unsqueeze = None
        batch_norm_2 = torch.nn.functional.batch_norm(
            x_1,
            l_self_modules_backbone_modules_block1_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block1_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block1_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block1_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block1_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block1_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_norm1_parameters_bias_ = (None)
        shorcut = batch_norm_2.clone()
        x_2 = torch.conv2d(
            batch_norm_2,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_2 = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_3 = torch._C._nn.gelu(x_2, approximate="none")
        x_2 = None
        u = x_3.clone()
        attn = torch.conv2d(
            x_3,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            64,
        )
        x_3 = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_0 = torch.conv2d(
            attn,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_1 = torch.conv2d(
            attn_0,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            64,
        )
        attn_0 = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_2 = torch.conv2d(
            attn,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_3 = torch.conv2d(
            attn_2,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            64,
        )
        attn_2 = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_4 = torch.conv2d(
            attn,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_5 = torch.conv2d(
            attn_4,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            64,
        )
        attn_4 = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add = attn + attn_1
        attn = attn_1 = None
        add_1 = add + attn_3
        add = attn_3 = None
        attn_6 = add_1 + attn_5
        add_1 = attn_5 = None
        attn_7 = torch.conv2d(
            attn_6,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_6 = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_4 = attn_7 * u
        attn_7 = u = None
        x_5 = torch.conv2d(
            x_4,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_4 = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_6 = x_5 + shorcut
        x_5 = shorcut = None
        mul_1 = unsqueeze_1 * x_6
        unsqueeze_1 = x_6 = None
        x_7 = x_1 + mul_1
        x_1 = mul_1 = None
        unsqueeze_2 = l_self_modules_backbone_modules_block1_modules_0_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block1_modules_0_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_3 = unsqueeze_2.unsqueeze(-1)
        unsqueeze_2 = None
        batch_norm_3 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_backbone_modules_block1_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block1_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block1_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block1_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block1_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block1_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_norm2_parameters_bias_ = (None)
        x_8 = torch.conv2d(
            batch_norm_3,
            l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_3 = l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_8 = l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_10 = torch._C._nn.gelu(x_9, approximate="none")
        x_9 = None
        x_11 = torch.nn.functional.dropout(x_10, 0.0, False, False)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_13 = torch.nn.functional.dropout(x_12, 0.0, False, False)
        x_12 = None
        mul_2 = unsqueeze_3 * x_13
        unsqueeze_3 = x_13 = None
        x_14 = x_7 + mul_2
        x_7 = mul_2 = None
        view_1 = x_14.view(1, 64, 16384)
        x_14 = None
        x_15 = view_1.permute(0, 2, 1)
        view_1 = None
        permute_2 = x_15.permute(0, 2, 1)
        x_15 = None
        x_16 = permute_2.view(1, 64, 128, 128)
        permute_2 = None
        unsqueeze_4 = l_self_modules_backbone_modules_block1_modules_1_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block1_modules_1_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_5 = unsqueeze_4.unsqueeze(-1)
        unsqueeze_4 = None
        batch_norm_4 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_backbone_modules_block1_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block1_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block1_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block1_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block1_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block1_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_norm1_parameters_bias_ = (None)
        shorcut_1 = batch_norm_4.clone()
        x_17 = torch.conv2d(
            batch_norm_4,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_4 = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_18 = torch._C._nn.gelu(x_17, approximate="none")
        x_17 = None
        u_1 = x_18.clone()
        attn_8 = torch.conv2d(
            x_18,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            64,
        )
        x_18 = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_9 = torch.conv2d(
            attn_8,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_10 = torch.conv2d(
            attn_9,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            64,
        )
        attn_9 = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_11 = torch.conv2d(
            attn_8,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_12 = torch.conv2d(
            attn_11,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            64,
        )
        attn_11 = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_13 = torch.conv2d(
            attn_8,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_14 = torch.conv2d(
            attn_13,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            64,
        )
        attn_13 = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_6 = attn_8 + attn_10
        attn_8 = attn_10 = None
        add_7 = add_6 + attn_12
        add_6 = attn_12 = None
        attn_15 = add_7 + attn_14
        add_7 = attn_14 = None
        attn_16 = torch.conv2d(
            attn_15,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_15 = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_19 = attn_16 * u_1
        attn_16 = u_1 = None
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_19 = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_21 = x_20 + shorcut_1
        x_20 = shorcut_1 = None
        mul_4 = unsqueeze_5 * x_21
        unsqueeze_5 = x_21 = None
        x_22 = x_16 + mul_4
        x_16 = mul_4 = None
        unsqueeze_6 = l_self_modules_backbone_modules_block1_modules_1_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block1_modules_1_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_7 = unsqueeze_6.unsqueeze(-1)
        unsqueeze_6 = None
        batch_norm_5 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_backbone_modules_block1_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block1_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block1_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block1_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block1_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block1_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_norm2_parameters_bias_ = (None)
        x_23 = torch.conv2d(
            batch_norm_5,
            l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_5 = l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_23 = l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_25 = torch._C._nn.gelu(x_24, approximate="none")
        x_24 = None
        x_26 = torch.nn.functional.dropout(x_25, 0.0, False, False)
        x_25 = None
        x_27 = torch.conv2d(
            x_26,
            l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_28 = torch.nn.functional.dropout(x_27, 0.0, False, False)
        x_27 = None
        mul_5 = unsqueeze_7 * x_28
        unsqueeze_7 = x_28 = None
        x_29 = x_22 + mul_5
        x_22 = mul_5 = None
        view_3 = x_29.view(1, 64, 16384)
        x_29 = None
        x_30 = view_3.permute(0, 2, 1)
        view_3 = None
        permute_4 = x_30.permute(0, 2, 1)
        x_30 = None
        x_31 = permute_4.view(1, 64, 128, 128)
        permute_4 = None
        unsqueeze_8 = l_self_modules_backbone_modules_block1_modules_2_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block1_modules_2_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_9 = unsqueeze_8.unsqueeze(-1)
        unsqueeze_8 = None
        batch_norm_6 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_backbone_modules_block1_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block1_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block1_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block1_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block1_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block1_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_norm1_parameters_bias_ = (None)
        shorcut_2 = batch_norm_6.clone()
        x_32 = torch.conv2d(
            batch_norm_6,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_6 = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_33 = torch._C._nn.gelu(x_32, approximate="none")
        x_32 = None
        u_2 = x_33.clone()
        attn_17 = torch.conv2d(
            x_33,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            64,
        )
        x_33 = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_18 = torch.conv2d(
            attn_17,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_19 = torch.conv2d(
            attn_18,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            64,
        )
        attn_18 = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_20 = torch.conv2d(
            attn_17,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_21 = torch.conv2d(
            attn_20,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            64,
        )
        attn_20 = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_22 = torch.conv2d(
            attn_17,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_23 = torch.conv2d(
            attn_22,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            64,
        )
        attn_22 = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_12 = attn_17 + attn_19
        attn_17 = attn_19 = None
        add_13 = add_12 + attn_21
        add_12 = attn_21 = None
        attn_24 = add_13 + attn_23
        add_13 = attn_23 = None
        attn_25 = torch.conv2d(
            attn_24,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_24 = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_34 = attn_25 * u_2
        attn_25 = u_2 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_36 = x_35 + shorcut_2
        x_35 = shorcut_2 = None
        mul_7 = unsqueeze_9 * x_36
        unsqueeze_9 = x_36 = None
        x_37 = x_31 + mul_7
        x_31 = mul_7 = None
        unsqueeze_10 = l_self_modules_backbone_modules_block1_modules_2_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block1_modules_2_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_11 = unsqueeze_10.unsqueeze(-1)
        unsqueeze_10 = None
        batch_norm_7 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_backbone_modules_block1_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block1_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block1_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block1_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block1_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block1_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_norm2_parameters_bias_ = (None)
        x_38 = torch.conv2d(
            batch_norm_7,
            l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_7 = l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_38 = l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_40 = torch._C._nn.gelu(x_39, approximate="none")
        x_39 = None
        x_41 = torch.nn.functional.dropout(x_40, 0.0, False, False)
        x_40 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_41 = l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block1_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_43 = torch.nn.functional.dropout(x_42, 0.0, False, False)
        x_42 = None
        mul_8 = unsqueeze_11 * x_43
        unsqueeze_11 = x_43 = None
        x_44 = x_37 + mul_8
        x_37 = mul_8 = None
        view_5 = x_44.view(1, 64, 16384)
        x_44 = None
        x_45 = view_5.permute(0, 2, 1)
        view_5 = None
        x_46 = torch.nn.functional.layer_norm(
            x_45,
            (64,),
            l_self_modules_backbone_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_45 = (
            l_self_modules_backbone_modules_norm1_parameters_weight_
        ) = l_self_modules_backbone_modules_norm1_parameters_bias_ = None
        reshape = x_46.reshape(1, 128, 128, -1)
        x_46 = None
        permute_6 = reshape.permute(0, 3, 1, 2)
        reshape = None
        x_47 = permute_6.contiguous()
        permute_6 = None
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_backbone_modules_patch_embed2_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed2_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_47 = (
            l_self_modules_backbone_modules_patch_embed2_modules_proj_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_patch_embed2_modules_proj_parameters_bias_
        ) = None
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_backbone_modules_patch_embed2_modules_norm_buffers_running_mean_,
            l_self_modules_backbone_modules_patch_embed2_modules_norm_buffers_running_var_,
            l_self_modules_backbone_modules_patch_embed2_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = l_self_modules_backbone_modules_patch_embed2_modules_norm_buffers_running_mean_ = l_self_modules_backbone_modules_patch_embed2_modules_norm_buffers_running_var_ = (
            l_self_modules_backbone_modules_patch_embed2_modules_norm_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_patch_embed2_modules_norm_parameters_bias_
        ) = None
        flatten_1 = x_49.flatten(2)
        x_49 = None
        x_50 = flatten_1.transpose(1, 2)
        flatten_1 = None
        permute_7 = x_50.permute(0, 2, 1)
        x_50 = None
        x_51 = permute_7.view(1, 128, 64, 64)
        permute_7 = None
        unsqueeze_12 = l_self_modules_backbone_modules_block2_modules_0_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block2_modules_0_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_13 = unsqueeze_12.unsqueeze(-1)
        unsqueeze_12 = None
        batch_norm_9 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block2_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block2_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block2_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_norm1_parameters_bias_ = (None)
        shorcut_3 = batch_norm_9.clone()
        x_52 = torch.conv2d(
            batch_norm_9,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_9 = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_53 = torch._C._nn.gelu(x_52, approximate="none")
        x_52 = None
        u_3 = x_53.clone()
        attn_26 = torch.conv2d(
            x_53,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            128,
        )
        x_53 = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_27 = torch.conv2d(
            attn_26,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_28 = torch.conv2d(
            attn_27,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            128,
        )
        attn_27 = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_29 = torch.conv2d(
            attn_26,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_30 = torch.conv2d(
            attn_29,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            128,
        )
        attn_29 = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_31 = torch.conv2d(
            attn_26,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_32 = torch.conv2d(
            attn_31,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            128,
        )
        attn_31 = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_18 = attn_26 + attn_28
        attn_26 = attn_28 = None
        add_19 = add_18 + attn_30
        add_18 = attn_30 = None
        attn_33 = add_19 + attn_32
        add_19 = attn_32 = None
        attn_34 = torch.conv2d(
            attn_33,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_33 = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_54 = attn_34 * u_3
        attn_34 = u_3 = None
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_54 = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_56 = x_55 + shorcut_3
        x_55 = shorcut_3 = None
        mul_10 = unsqueeze_13 * x_56
        unsqueeze_13 = x_56 = None
        x_57 = x_51 + mul_10
        x_51 = mul_10 = None
        unsqueeze_14 = l_self_modules_backbone_modules_block2_modules_0_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block2_modules_0_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_15 = unsqueeze_14.unsqueeze(-1)
        unsqueeze_14 = None
        batch_norm_10 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block2_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block2_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block2_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_norm2_parameters_bias_ = (None)
        x_58 = torch.conv2d(
            batch_norm_10,
            l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_10 = l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_58 = l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_60 = torch._C._nn.gelu(x_59, approximate="none")
        x_59 = None
        x_61 = torch.nn.functional.dropout(x_60, 0.0, False, False)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_63 = torch.nn.functional.dropout(x_62, 0.0, False, False)
        x_62 = None
        mul_11 = unsqueeze_15 * x_63
        unsqueeze_15 = x_63 = None
        x_64 = x_57 + mul_11
        x_57 = mul_11 = None
        view_7 = x_64.view(1, 128, 4096)
        x_64 = None
        x_65 = view_7.permute(0, 2, 1)
        view_7 = None
        permute_9 = x_65.permute(0, 2, 1)
        x_65 = None
        x_66 = permute_9.view(1, 128, 64, 64)
        permute_9 = None
        unsqueeze_16 = l_self_modules_backbone_modules_block2_modules_1_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block2_modules_1_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_17 = unsqueeze_16.unsqueeze(-1)
        unsqueeze_16 = None
        batch_norm_11 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block2_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block2_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block2_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_norm1_parameters_bias_ = (None)
        shorcut_4 = batch_norm_11.clone()
        x_67 = torch.conv2d(
            batch_norm_11,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_11 = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_68 = torch._C._nn.gelu(x_67, approximate="none")
        x_67 = None
        u_4 = x_68.clone()
        attn_35 = torch.conv2d(
            x_68,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            128,
        )
        x_68 = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_36 = torch.conv2d(
            attn_35,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_37 = torch.conv2d(
            attn_36,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            128,
        )
        attn_36 = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_38 = torch.conv2d(
            attn_35,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_39 = torch.conv2d(
            attn_38,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            128,
        )
        attn_38 = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_40 = torch.conv2d(
            attn_35,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_41 = torch.conv2d(
            attn_40,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            128,
        )
        attn_40 = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_24 = attn_35 + attn_37
        attn_35 = attn_37 = None
        add_25 = add_24 + attn_39
        add_24 = attn_39 = None
        attn_42 = add_25 + attn_41
        add_25 = attn_41 = None
        attn_43 = torch.conv2d(
            attn_42,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_42 = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_69 = attn_43 * u_4
        attn_43 = u_4 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_71 = x_70 + shorcut_4
        x_70 = shorcut_4 = None
        mul_13 = unsqueeze_17 * x_71
        unsqueeze_17 = x_71 = None
        x_72 = x_66 + mul_13
        x_66 = mul_13 = None
        unsqueeze_18 = l_self_modules_backbone_modules_block2_modules_1_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block2_modules_1_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_19 = unsqueeze_18.unsqueeze(-1)
        unsqueeze_18 = None
        batch_norm_12 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block2_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block2_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block2_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_norm2_parameters_bias_ = (None)
        x_73 = torch.conv2d(
            batch_norm_12,
            l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_12 = l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_74 = torch.conv2d(
            x_73,
            l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_73 = l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_75 = torch._C._nn.gelu(x_74, approximate="none")
        x_74 = None
        x_76 = torch.nn.functional.dropout(x_75, 0.0, False, False)
        x_75 = None
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_76 = l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_78 = torch.nn.functional.dropout(x_77, 0.0, False, False)
        x_77 = None
        mul_14 = unsqueeze_19 * x_78
        unsqueeze_19 = x_78 = None
        x_79 = x_72 + mul_14
        x_72 = mul_14 = None
        view_9 = x_79.view(1, 128, 4096)
        x_79 = None
        x_80 = view_9.permute(0, 2, 1)
        view_9 = None
        permute_11 = x_80.permute(0, 2, 1)
        x_80 = None
        x_81 = permute_11.view(1, 128, 64, 64)
        permute_11 = None
        unsqueeze_20 = l_self_modules_backbone_modules_block2_modules_2_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block2_modules_2_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_21 = unsqueeze_20.unsqueeze(-1)
        unsqueeze_20 = None
        batch_norm_13 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_backbone_modules_block2_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block2_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block2_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block2_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block2_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block2_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_norm1_parameters_bias_ = (None)
        shorcut_5 = batch_norm_13.clone()
        x_82 = torch.conv2d(
            batch_norm_13,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_13 = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_83 = torch._C._nn.gelu(x_82, approximate="none")
        x_82 = None
        u_5 = x_83.clone()
        attn_44 = torch.conv2d(
            x_83,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            128,
        )
        x_83 = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_45 = torch.conv2d(
            attn_44,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_46 = torch.conv2d(
            attn_45,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            128,
        )
        attn_45 = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_47 = torch.conv2d(
            attn_44,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_48 = torch.conv2d(
            attn_47,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            128,
        )
        attn_47 = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_49 = torch.conv2d(
            attn_44,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_50 = torch.conv2d(
            attn_49,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            128,
        )
        attn_49 = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_30 = attn_44 + attn_46
        attn_44 = attn_46 = None
        add_31 = add_30 + attn_48
        add_30 = attn_48 = None
        attn_51 = add_31 + attn_50
        add_31 = attn_50 = None
        attn_52 = torch.conv2d(
            attn_51,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_51 = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_84 = attn_52 * u_5
        attn_52 = u_5 = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_84 = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_86 = x_85 + shorcut_5
        x_85 = shorcut_5 = None
        mul_16 = unsqueeze_21 * x_86
        unsqueeze_21 = x_86 = None
        x_87 = x_81 + mul_16
        x_81 = mul_16 = None
        unsqueeze_22 = l_self_modules_backbone_modules_block2_modules_2_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block2_modules_2_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_23 = unsqueeze_22.unsqueeze(-1)
        unsqueeze_22 = None
        batch_norm_14 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_backbone_modules_block2_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block2_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block2_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block2_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block2_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block2_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_norm2_parameters_bias_ = (None)
        x_88 = torch.conv2d(
            batch_norm_14,
            l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_14 = l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_88 = l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_90 = torch._C._nn.gelu(x_89, approximate="none")
        x_89 = None
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_93 = torch.nn.functional.dropout(x_92, 0.0, False, False)
        x_92 = None
        mul_17 = unsqueeze_23 * x_93
        unsqueeze_23 = x_93 = None
        x_94 = x_87 + mul_17
        x_87 = mul_17 = None
        view_11 = x_94.view(1, 128, 4096)
        x_94 = None
        x_95 = view_11.permute(0, 2, 1)
        view_11 = None
        permute_13 = x_95.permute(0, 2, 1)
        x_95 = None
        x_96 = permute_13.view(1, 128, 64, 64)
        permute_13 = None
        unsqueeze_24 = l_self_modules_backbone_modules_block2_modules_3_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block2_modules_3_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_25 = unsqueeze_24.unsqueeze(-1)
        unsqueeze_24 = None
        batch_norm_15 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_backbone_modules_block2_modules_3_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block2_modules_3_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block2_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block2_modules_3_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block2_modules_3_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block2_modules_3_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_norm1_parameters_bias_ = (None)
        shorcut_6 = batch_norm_15.clone()
        x_97 = torch.conv2d(
            batch_norm_15,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_15 = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_98 = torch._C._nn.gelu(x_97, approximate="none")
        x_97 = None
        u_6 = x_98.clone()
        attn_53 = torch.conv2d(
            x_98,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            128,
        )
        x_98 = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_54 = torch.conv2d(
            attn_53,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_55 = torch.conv2d(
            attn_54,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            128,
        )
        attn_54 = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_56 = torch.conv2d(
            attn_53,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_57 = torch.conv2d(
            attn_56,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            128,
        )
        attn_56 = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_58 = torch.conv2d(
            attn_53,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_59 = torch.conv2d(
            attn_58,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            128,
        )
        attn_58 = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_36 = attn_53 + attn_55
        attn_53 = attn_55 = None
        add_37 = add_36 + attn_57
        add_36 = attn_57 = None
        attn_60 = add_37 + attn_59
        add_37 = attn_59 = None
        attn_61 = torch.conv2d(
            attn_60,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_60 = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_99 = attn_61 * u_6
        attn_61 = u_6 = None
        x_100 = torch.conv2d(
            x_99,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_99 = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_101 = x_100 + shorcut_6
        x_100 = shorcut_6 = None
        mul_19 = unsqueeze_25 * x_101
        unsqueeze_25 = x_101 = None
        x_102 = x_96 + mul_19
        x_96 = mul_19 = None
        unsqueeze_26 = l_self_modules_backbone_modules_block2_modules_3_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block2_modules_3_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_27 = unsqueeze_26.unsqueeze(-1)
        unsqueeze_26 = None
        batch_norm_16 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_backbone_modules_block2_modules_3_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block2_modules_3_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block2_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block2_modules_3_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block2_modules_3_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block2_modules_3_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_norm2_parameters_bias_ = (None)
        x_103 = torch.conv2d(
            batch_norm_16,
            l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_16 = l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_103 = l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_105 = torch._C._nn.gelu(x_104, approximate="none")
        x_104 = None
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_108 = torch.nn.functional.dropout(x_107, 0.0, False, False)
        x_107 = None
        mul_20 = unsqueeze_27 * x_108
        unsqueeze_27 = x_108 = None
        x_109 = x_102 + mul_20
        x_102 = mul_20 = None
        view_13 = x_109.view(1, 128, 4096)
        x_109 = None
        x_110 = view_13.permute(0, 2, 1)
        view_13 = None
        permute_15 = x_110.permute(0, 2, 1)
        x_110 = None
        x_111 = permute_15.view(1, 128, 64, 64)
        permute_15 = None
        unsqueeze_28 = l_self_modules_backbone_modules_block2_modules_4_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block2_modules_4_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_29 = unsqueeze_28.unsqueeze(-1)
        unsqueeze_28 = None
        batch_norm_17 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_backbone_modules_block2_modules_4_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block2_modules_4_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block2_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block2_modules_4_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block2_modules_4_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block2_modules_4_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_norm1_parameters_bias_ = (None)
        shorcut_7 = batch_norm_17.clone()
        x_112 = torch.conv2d(
            batch_norm_17,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_17 = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_113 = torch._C._nn.gelu(x_112, approximate="none")
        x_112 = None
        u_7 = x_113.clone()
        attn_62 = torch.conv2d(
            x_113,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            128,
        )
        x_113 = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_63 = torch.conv2d(
            attn_62,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_64 = torch.conv2d(
            attn_63,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            128,
        )
        attn_63 = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_65 = torch.conv2d(
            attn_62,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_66 = torch.conv2d(
            attn_65,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            128,
        )
        attn_65 = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_67 = torch.conv2d(
            attn_62,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_68 = torch.conv2d(
            attn_67,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            128,
        )
        attn_67 = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_42 = attn_62 + attn_64
        attn_62 = attn_64 = None
        add_43 = add_42 + attn_66
        add_42 = attn_66 = None
        attn_69 = add_43 + attn_68
        add_43 = attn_68 = None
        attn_70 = torch.conv2d(
            attn_69,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_69 = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_114 = attn_70 * u_7
        attn_70 = u_7 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_114 = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_116 = x_115 + shorcut_7
        x_115 = shorcut_7 = None
        mul_22 = unsqueeze_29 * x_116
        unsqueeze_29 = x_116 = None
        x_117 = x_111 + mul_22
        x_111 = mul_22 = None
        unsqueeze_30 = l_self_modules_backbone_modules_block2_modules_4_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block2_modules_4_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_31 = unsqueeze_30.unsqueeze(-1)
        unsqueeze_30 = None
        batch_norm_18 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_backbone_modules_block2_modules_4_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block2_modules_4_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block2_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block2_modules_4_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block2_modules_4_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block2_modules_4_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_norm2_parameters_bias_ = (None)
        x_118 = torch.conv2d(
            batch_norm_18,
            l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_18 = l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_118 = l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_120 = torch._C._nn.gelu(x_119, approximate="none")
        x_119 = None
        x_121 = torch.nn.functional.dropout(x_120, 0.0, False, False)
        x_120 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_123 = torch.nn.functional.dropout(x_122, 0.0, False, False)
        x_122 = None
        mul_23 = unsqueeze_31 * x_123
        unsqueeze_31 = x_123 = None
        x_124 = x_117 + mul_23
        x_117 = mul_23 = None
        view_15 = x_124.view(1, 128, 4096)
        x_124 = None
        x_125 = view_15.permute(0, 2, 1)
        view_15 = None
        x_126 = torch.nn.functional.layer_norm(
            x_125,
            (128,),
            l_self_modules_backbone_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_125 = (
            l_self_modules_backbone_modules_norm2_parameters_weight_
        ) = l_self_modules_backbone_modules_norm2_parameters_bias_ = None
        reshape_1 = x_126.reshape(1, 64, 64, -1)
        x_126 = None
        permute_17 = reshape_1.permute(0, 3, 1, 2)
        reshape_1 = None
        x_127 = permute_17.contiguous()
        permute_17 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_backbone_modules_patch_embed3_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed3_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_patch_embed3_modules_proj_parameters_weight_ = (
            l_self_modules_backbone_modules_patch_embed3_modules_proj_parameters_bias_
        ) = None
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_backbone_modules_patch_embed3_modules_norm_buffers_running_mean_,
            l_self_modules_backbone_modules_patch_embed3_modules_norm_buffers_running_var_,
            l_self_modules_backbone_modules_patch_embed3_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_backbone_modules_patch_embed3_modules_norm_buffers_running_mean_ = l_self_modules_backbone_modules_patch_embed3_modules_norm_buffers_running_var_ = (
            l_self_modules_backbone_modules_patch_embed3_modules_norm_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_patch_embed3_modules_norm_parameters_bias_
        ) = None
        flatten_2 = x_129.flatten(2)
        x_129 = None
        x_130 = flatten_2.transpose(1, 2)
        flatten_2 = None
        permute_18 = x_130.permute(0, 2, 1)
        x_130 = None
        x_131 = permute_18.view(1, 320, 32, 32)
        permute_18 = None
        unsqueeze_32 = l_self_modules_backbone_modules_block3_modules_0_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_0_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_33 = unsqueeze_32.unsqueeze(-1)
        unsqueeze_32 = None
        batch_norm_20 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_norm1_parameters_bias_ = (None)
        shorcut_8 = batch_norm_20.clone()
        x_132 = torch.conv2d(
            batch_norm_20,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_20 = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_133 = torch._C._nn.gelu(x_132, approximate="none")
        x_132 = None
        u_8 = x_133.clone()
        attn_71 = torch.conv2d(
            x_133,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_133 = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_72 = torch.conv2d(
            attn_71,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_73 = torch.conv2d(
            attn_72,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_72 = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_74 = torch.conv2d(
            attn_71,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_75 = torch.conv2d(
            attn_74,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_74 = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_76 = torch.conv2d(
            attn_71,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_77 = torch.conv2d(
            attn_76,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_76 = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_48 = attn_71 + attn_73
        attn_71 = attn_73 = None
        add_49 = add_48 + attn_75
        add_48 = attn_75 = None
        attn_78 = add_49 + attn_77
        add_49 = attn_77 = None
        attn_79 = torch.conv2d(
            attn_78,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_78 = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_134 = attn_79 * u_8
        attn_79 = u_8 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_134 = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_136 = x_135 + shorcut_8
        x_135 = shorcut_8 = None
        mul_25 = unsqueeze_33 * x_136
        unsqueeze_33 = x_136 = None
        x_137 = x_131 + mul_25
        x_131 = mul_25 = None
        unsqueeze_34 = l_self_modules_backbone_modules_block3_modules_0_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_0_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_35 = unsqueeze_34.unsqueeze(-1)
        unsqueeze_34 = None
        batch_norm_21 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_norm2_parameters_bias_ = (None)
        x_138 = torch.conv2d(
            batch_norm_21,
            l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_21 = l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_138 = l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_140 = torch._C._nn.gelu(x_139, approximate="none")
        x_139 = None
        x_141 = torch.nn.functional.dropout(x_140, 0.0, False, False)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_143 = torch.nn.functional.dropout(x_142, 0.0, False, False)
        x_142 = None
        mul_26 = unsqueeze_35 * x_143
        unsqueeze_35 = x_143 = None
        x_144 = x_137 + mul_26
        x_137 = mul_26 = None
        view_17 = x_144.view(1, 320, 1024)
        x_144 = None
        x_145 = view_17.permute(0, 2, 1)
        view_17 = None
        permute_20 = x_145.permute(0, 2, 1)
        x_145 = None
        x_146 = permute_20.view(1, 320, 32, 32)
        permute_20 = None
        unsqueeze_36 = l_self_modules_backbone_modules_block3_modules_1_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_1_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_37 = unsqueeze_36.unsqueeze(-1)
        unsqueeze_36 = None
        batch_norm_22 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_norm1_parameters_bias_ = (None)
        shorcut_9 = batch_norm_22.clone()
        x_147 = torch.conv2d(
            batch_norm_22,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_22 = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_148 = torch._C._nn.gelu(x_147, approximate="none")
        x_147 = None
        u_9 = x_148.clone()
        attn_80 = torch.conv2d(
            x_148,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_148 = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_81 = torch.conv2d(
            attn_80,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_82 = torch.conv2d(
            attn_81,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_81 = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_83 = torch.conv2d(
            attn_80,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_84 = torch.conv2d(
            attn_83,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_83 = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_85 = torch.conv2d(
            attn_80,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_86 = torch.conv2d(
            attn_85,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_85 = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_54 = attn_80 + attn_82
        attn_80 = attn_82 = None
        add_55 = add_54 + attn_84
        add_54 = attn_84 = None
        attn_87 = add_55 + attn_86
        add_55 = attn_86 = None
        attn_88 = torch.conv2d(
            attn_87,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_87 = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_149 = attn_88 * u_9
        attn_88 = u_9 = None
        x_150 = torch.conv2d(
            x_149,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_149 = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_151 = x_150 + shorcut_9
        x_150 = shorcut_9 = None
        mul_28 = unsqueeze_37 * x_151
        unsqueeze_37 = x_151 = None
        x_152 = x_146 + mul_28
        x_146 = mul_28 = None
        unsqueeze_38 = l_self_modules_backbone_modules_block3_modules_1_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_1_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_39 = unsqueeze_38.unsqueeze(-1)
        unsqueeze_38 = None
        batch_norm_23 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_norm2_parameters_bias_ = (None)
        x_153 = torch.conv2d(
            batch_norm_23,
            l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_23 = l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_153 = l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_155 = torch._C._nn.gelu(x_154, approximate="none")
        x_154 = None
        x_156 = torch.nn.functional.dropout(x_155, 0.0, False, False)
        x_155 = None
        x_157 = torch.conv2d(
            x_156,
            l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_156 = l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_158 = torch.nn.functional.dropout(x_157, 0.0, False, False)
        x_157 = None
        mul_29 = unsqueeze_39 * x_158
        unsqueeze_39 = x_158 = None
        x_159 = x_152 + mul_29
        x_152 = mul_29 = None
        view_19 = x_159.view(1, 320, 1024)
        x_159 = None
        x_160 = view_19.permute(0, 2, 1)
        view_19 = None
        permute_22 = x_160.permute(0, 2, 1)
        x_160 = None
        x_161 = permute_22.view(1, 320, 32, 32)
        permute_22 = None
        unsqueeze_40 = l_self_modules_backbone_modules_block3_modules_2_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_2_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_41 = unsqueeze_40.unsqueeze(-1)
        unsqueeze_40 = None
        batch_norm_24 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_norm1_parameters_bias_ = (None)
        shorcut_10 = batch_norm_24.clone()
        x_162 = torch.conv2d(
            batch_norm_24,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_24 = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_163 = torch._C._nn.gelu(x_162, approximate="none")
        x_162 = None
        u_10 = x_163.clone()
        attn_89 = torch.conv2d(
            x_163,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_163 = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_90 = torch.conv2d(
            attn_89,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_91 = torch.conv2d(
            attn_90,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_90 = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_92 = torch.conv2d(
            attn_89,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_93 = torch.conv2d(
            attn_92,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_92 = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_94 = torch.conv2d(
            attn_89,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_95 = torch.conv2d(
            attn_94,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_94 = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_60 = attn_89 + attn_91
        attn_89 = attn_91 = None
        add_61 = add_60 + attn_93
        add_60 = attn_93 = None
        attn_96 = add_61 + attn_95
        add_61 = attn_95 = None
        attn_97 = torch.conv2d(
            attn_96,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_96 = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_164 = attn_97 * u_10
        attn_97 = u_10 = None
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_164 = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_166 = x_165 + shorcut_10
        x_165 = shorcut_10 = None
        mul_31 = unsqueeze_41 * x_166
        unsqueeze_41 = x_166 = None
        x_167 = x_161 + mul_31
        x_161 = mul_31 = None
        unsqueeze_42 = l_self_modules_backbone_modules_block3_modules_2_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_2_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_43 = unsqueeze_42.unsqueeze(-1)
        unsqueeze_42 = None
        batch_norm_25 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_norm2_parameters_bias_ = (None)
        x_168 = torch.conv2d(
            batch_norm_25,
            l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_25 = l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_169 = torch.conv2d(
            x_168,
            l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_168 = l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_170 = torch._C._nn.gelu(x_169, approximate="none")
        x_169 = None
        x_171 = torch.nn.functional.dropout(x_170, 0.0, False, False)
        x_170 = None
        x_172 = torch.conv2d(
            x_171,
            l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_171 = l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_173 = torch.nn.functional.dropout(x_172, 0.0, False, False)
        x_172 = None
        mul_32 = unsqueeze_43 * x_173
        unsqueeze_43 = x_173 = None
        x_174 = x_167 + mul_32
        x_167 = mul_32 = None
        view_21 = x_174.view(1, 320, 1024)
        x_174 = None
        x_175 = view_21.permute(0, 2, 1)
        view_21 = None
        permute_24 = x_175.permute(0, 2, 1)
        x_175 = None
        x_176 = permute_24.view(1, 320, 32, 32)
        permute_24 = None
        unsqueeze_44 = l_self_modules_backbone_modules_block3_modules_3_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_3_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_45 = unsqueeze_44.unsqueeze(-1)
        unsqueeze_44 = None
        batch_norm_26 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_3_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_3_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_3_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_norm1_parameters_bias_ = (None)
        shorcut_11 = batch_norm_26.clone()
        x_177 = torch.conv2d(
            batch_norm_26,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_26 = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_178 = torch._C._nn.gelu(x_177, approximate="none")
        x_177 = None
        u_11 = x_178.clone()
        attn_98 = torch.conv2d(
            x_178,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_178 = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_99 = torch.conv2d(
            attn_98,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_100 = torch.conv2d(
            attn_99,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_99 = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_101 = torch.conv2d(
            attn_98,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_102 = torch.conv2d(
            attn_101,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_101 = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_103 = torch.conv2d(
            attn_98,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_104 = torch.conv2d(
            attn_103,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_103 = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_66 = attn_98 + attn_100
        attn_98 = attn_100 = None
        add_67 = add_66 + attn_102
        add_66 = attn_102 = None
        attn_105 = add_67 + attn_104
        add_67 = attn_104 = None
        attn_106 = torch.conv2d(
            attn_105,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_105 = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_179 = attn_106 * u_11
        attn_106 = u_11 = None
        x_180 = torch.conv2d(
            x_179,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_179 = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_181 = x_180 + shorcut_11
        x_180 = shorcut_11 = None
        mul_34 = unsqueeze_45 * x_181
        unsqueeze_45 = x_181 = None
        x_182 = x_176 + mul_34
        x_176 = mul_34 = None
        unsqueeze_46 = l_self_modules_backbone_modules_block3_modules_3_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_3_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_47 = unsqueeze_46.unsqueeze(-1)
        unsqueeze_46 = None
        batch_norm_27 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_3_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_3_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_3_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_norm2_parameters_bias_ = (None)
        x_183 = torch.conv2d(
            batch_norm_27,
            l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_27 = l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_184 = torch.conv2d(
            x_183,
            l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_183 = l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_185 = torch._C._nn.gelu(x_184, approximate="none")
        x_184 = None
        x_186 = torch.nn.functional.dropout(x_185, 0.0, False, False)
        x_185 = None
        x_187 = torch.conv2d(
            x_186,
            l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_186 = l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_188 = torch.nn.functional.dropout(x_187, 0.0, False, False)
        x_187 = None
        mul_35 = unsqueeze_47 * x_188
        unsqueeze_47 = x_188 = None
        x_189 = x_182 + mul_35
        x_182 = mul_35 = None
        view_23 = x_189.view(1, 320, 1024)
        x_189 = None
        x_190 = view_23.permute(0, 2, 1)
        view_23 = None
        permute_26 = x_190.permute(0, 2, 1)
        x_190 = None
        x_191 = permute_26.view(1, 320, 32, 32)
        permute_26 = None
        unsqueeze_48 = l_self_modules_backbone_modules_block3_modules_4_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_4_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_49 = unsqueeze_48.unsqueeze(-1)
        unsqueeze_48 = None
        batch_norm_28 = torch.nn.functional.batch_norm(
            x_191,
            l_self_modules_backbone_modules_block3_modules_4_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_4_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_4_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_4_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_4_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_norm1_parameters_bias_ = (None)
        shorcut_12 = batch_norm_28.clone()
        x_192 = torch.conv2d(
            batch_norm_28,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_28 = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_193 = torch._C._nn.gelu(x_192, approximate="none")
        x_192 = None
        u_12 = x_193.clone()
        attn_107 = torch.conv2d(
            x_193,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_193 = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_108 = torch.conv2d(
            attn_107,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_109 = torch.conv2d(
            attn_108,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_108 = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_110 = torch.conv2d(
            attn_107,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_111 = torch.conv2d(
            attn_110,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_110 = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_112 = torch.conv2d(
            attn_107,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_113 = torch.conv2d(
            attn_112,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_112 = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_72 = attn_107 + attn_109
        attn_107 = attn_109 = None
        add_73 = add_72 + attn_111
        add_72 = attn_111 = None
        attn_114 = add_73 + attn_113
        add_73 = attn_113 = None
        attn_115 = torch.conv2d(
            attn_114,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_114 = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_194 = attn_115 * u_12
        attn_115 = u_12 = None
        x_195 = torch.conv2d(
            x_194,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_194 = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_196 = x_195 + shorcut_12
        x_195 = shorcut_12 = None
        mul_37 = unsqueeze_49 * x_196
        unsqueeze_49 = x_196 = None
        x_197 = x_191 + mul_37
        x_191 = mul_37 = None
        unsqueeze_50 = l_self_modules_backbone_modules_block3_modules_4_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_4_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_51 = unsqueeze_50.unsqueeze(-1)
        unsqueeze_50 = None
        batch_norm_29 = torch.nn.functional.batch_norm(
            x_197,
            l_self_modules_backbone_modules_block3_modules_4_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_4_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_4_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_4_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_4_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_norm2_parameters_bias_ = (None)
        x_198 = torch.conv2d(
            batch_norm_29,
            l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_29 = l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_199 = torch.conv2d(
            x_198,
            l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_198 = l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_200 = torch._C._nn.gelu(x_199, approximate="none")
        x_199 = None
        x_201 = torch.nn.functional.dropout(x_200, 0.0, False, False)
        x_200 = None
        x_202 = torch.conv2d(
            x_201,
            l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_201 = l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_203 = torch.nn.functional.dropout(x_202, 0.0, False, False)
        x_202 = None
        mul_38 = unsqueeze_51 * x_203
        unsqueeze_51 = x_203 = None
        x_204 = x_197 + mul_38
        x_197 = mul_38 = None
        view_25 = x_204.view(1, 320, 1024)
        x_204 = None
        x_205 = view_25.permute(0, 2, 1)
        view_25 = None
        permute_28 = x_205.permute(0, 2, 1)
        x_205 = None
        x_206 = permute_28.view(1, 320, 32, 32)
        permute_28 = None
        unsqueeze_52 = l_self_modules_backbone_modules_block3_modules_5_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_5_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_53 = unsqueeze_52.unsqueeze(-1)
        unsqueeze_52 = None
        batch_norm_30 = torch.nn.functional.batch_norm(
            x_206,
            l_self_modules_backbone_modules_block3_modules_5_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_5_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_5_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_5_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_5_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_norm1_parameters_bias_ = (None)
        shorcut_13 = batch_norm_30.clone()
        x_207 = torch.conv2d(
            batch_norm_30,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_30 = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_208 = torch._C._nn.gelu(x_207, approximate="none")
        x_207 = None
        u_13 = x_208.clone()
        attn_116 = torch.conv2d(
            x_208,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_208 = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_117 = torch.conv2d(
            attn_116,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_118 = torch.conv2d(
            attn_117,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_117 = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_119 = torch.conv2d(
            attn_116,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_120 = torch.conv2d(
            attn_119,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_119 = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_121 = torch.conv2d(
            attn_116,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_122 = torch.conv2d(
            attn_121,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_121 = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_78 = attn_116 + attn_118
        attn_116 = attn_118 = None
        add_79 = add_78 + attn_120
        add_78 = attn_120 = None
        attn_123 = add_79 + attn_122
        add_79 = attn_122 = None
        attn_124 = torch.conv2d(
            attn_123,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_123 = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_209 = attn_124 * u_13
        attn_124 = u_13 = None
        x_210 = torch.conv2d(
            x_209,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_209 = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_211 = x_210 + shorcut_13
        x_210 = shorcut_13 = None
        mul_40 = unsqueeze_53 * x_211
        unsqueeze_53 = x_211 = None
        x_212 = x_206 + mul_40
        x_206 = mul_40 = None
        unsqueeze_54 = l_self_modules_backbone_modules_block3_modules_5_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_5_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_55 = unsqueeze_54.unsqueeze(-1)
        unsqueeze_54 = None
        batch_norm_31 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_backbone_modules_block3_modules_5_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_5_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_5_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_5_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_5_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_norm2_parameters_bias_ = (None)
        x_213 = torch.conv2d(
            batch_norm_31,
            l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_31 = l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_214 = torch.conv2d(
            x_213,
            l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_213 = l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_215 = torch._C._nn.gelu(x_214, approximate="none")
        x_214 = None
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        x_217 = torch.conv2d(
            x_216,
            l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_216 = l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_218 = torch.nn.functional.dropout(x_217, 0.0, False, False)
        x_217 = None
        mul_41 = unsqueeze_55 * x_218
        unsqueeze_55 = x_218 = None
        x_219 = x_212 + mul_41
        x_212 = mul_41 = None
        view_27 = x_219.view(1, 320, 1024)
        x_219 = None
        x_220 = view_27.permute(0, 2, 1)
        view_27 = None
        permute_30 = x_220.permute(0, 2, 1)
        x_220 = None
        x_221 = permute_30.view(1, 320, 32, 32)
        permute_30 = None
        unsqueeze_56 = l_self_modules_backbone_modules_block3_modules_6_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_6_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_57 = unsqueeze_56.unsqueeze(-1)
        unsqueeze_56 = None
        batch_norm_32 = torch.nn.functional.batch_norm(
            x_221,
            l_self_modules_backbone_modules_block3_modules_6_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_6_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_6_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_6_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_6_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_norm1_parameters_bias_ = (None)
        shorcut_14 = batch_norm_32.clone()
        x_222 = torch.conv2d(
            batch_norm_32,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_32 = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_223 = torch._C._nn.gelu(x_222, approximate="none")
        x_222 = None
        u_14 = x_223.clone()
        attn_125 = torch.conv2d(
            x_223,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_223 = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_126 = torch.conv2d(
            attn_125,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_127 = torch.conv2d(
            attn_126,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_126 = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_128 = torch.conv2d(
            attn_125,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_129 = torch.conv2d(
            attn_128,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_128 = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_130 = torch.conv2d(
            attn_125,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_131 = torch.conv2d(
            attn_130,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_130 = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_84 = attn_125 + attn_127
        attn_125 = attn_127 = None
        add_85 = add_84 + attn_129
        add_84 = attn_129 = None
        attn_132 = add_85 + attn_131
        add_85 = attn_131 = None
        attn_133 = torch.conv2d(
            attn_132,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_132 = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_224 = attn_133 * u_14
        attn_133 = u_14 = None
        x_225 = torch.conv2d(
            x_224,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_224 = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_226 = x_225 + shorcut_14
        x_225 = shorcut_14 = None
        mul_43 = unsqueeze_57 * x_226
        unsqueeze_57 = x_226 = None
        x_227 = x_221 + mul_43
        x_221 = mul_43 = None
        unsqueeze_58 = l_self_modules_backbone_modules_block3_modules_6_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_6_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_59 = unsqueeze_58.unsqueeze(-1)
        unsqueeze_58 = None
        batch_norm_33 = torch.nn.functional.batch_norm(
            x_227,
            l_self_modules_backbone_modules_block3_modules_6_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_6_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_6_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_6_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_6_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_norm2_parameters_bias_ = (None)
        x_228 = torch.conv2d(
            batch_norm_33,
            l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_33 = l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_229 = torch.conv2d(
            x_228,
            l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_228 = l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_230 = torch._C._nn.gelu(x_229, approximate="none")
        x_229 = None
        x_231 = torch.nn.functional.dropout(x_230, 0.0, False, False)
        x_230 = None
        x_232 = torch.conv2d(
            x_231,
            l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_231 = l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_233 = torch.nn.functional.dropout(x_232, 0.0, False, False)
        x_232 = None
        mul_44 = unsqueeze_59 * x_233
        unsqueeze_59 = x_233 = None
        x_234 = x_227 + mul_44
        x_227 = mul_44 = None
        view_29 = x_234.view(1, 320, 1024)
        x_234 = None
        x_235 = view_29.permute(0, 2, 1)
        view_29 = None
        permute_32 = x_235.permute(0, 2, 1)
        x_235 = None
        x_236 = permute_32.view(1, 320, 32, 32)
        permute_32 = None
        unsqueeze_60 = l_self_modules_backbone_modules_block3_modules_7_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_7_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_61 = unsqueeze_60.unsqueeze(-1)
        unsqueeze_60 = None
        batch_norm_34 = torch.nn.functional.batch_norm(
            x_236,
            l_self_modules_backbone_modules_block3_modules_7_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_7_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_7_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_7_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_7_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_norm1_parameters_bias_ = (None)
        shorcut_15 = batch_norm_34.clone()
        x_237 = torch.conv2d(
            batch_norm_34,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_34 = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_238 = torch._C._nn.gelu(x_237, approximate="none")
        x_237 = None
        u_15 = x_238.clone()
        attn_134 = torch.conv2d(
            x_238,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_238 = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_135 = torch.conv2d(
            attn_134,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_136 = torch.conv2d(
            attn_135,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_135 = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_137 = torch.conv2d(
            attn_134,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_138 = torch.conv2d(
            attn_137,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_137 = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_139 = torch.conv2d(
            attn_134,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_140 = torch.conv2d(
            attn_139,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_139 = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_90 = attn_134 + attn_136
        attn_134 = attn_136 = None
        add_91 = add_90 + attn_138
        add_90 = attn_138 = None
        attn_141 = add_91 + attn_140
        add_91 = attn_140 = None
        attn_142 = torch.conv2d(
            attn_141,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_141 = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_239 = attn_142 * u_15
        attn_142 = u_15 = None
        x_240 = torch.conv2d(
            x_239,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_239 = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_241 = x_240 + shorcut_15
        x_240 = shorcut_15 = None
        mul_46 = unsqueeze_61 * x_241
        unsqueeze_61 = x_241 = None
        x_242 = x_236 + mul_46
        x_236 = mul_46 = None
        unsqueeze_62 = l_self_modules_backbone_modules_block3_modules_7_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_7_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_63 = unsqueeze_62.unsqueeze(-1)
        unsqueeze_62 = None
        batch_norm_35 = torch.nn.functional.batch_norm(
            x_242,
            l_self_modules_backbone_modules_block3_modules_7_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_7_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_7_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_7_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_7_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_norm2_parameters_bias_ = (None)
        x_243 = torch.conv2d(
            batch_norm_35,
            l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_35 = l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_243 = l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_245 = torch._C._nn.gelu(x_244, approximate="none")
        x_244 = None
        x_246 = torch.nn.functional.dropout(x_245, 0.0, False, False)
        x_245 = None
        x_247 = torch.conv2d(
            x_246,
            l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_246 = l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_248 = torch.nn.functional.dropout(x_247, 0.0, False, False)
        x_247 = None
        mul_47 = unsqueeze_63 * x_248
        unsqueeze_63 = x_248 = None
        x_249 = x_242 + mul_47
        x_242 = mul_47 = None
        view_31 = x_249.view(1, 320, 1024)
        x_249 = None
        x_250 = view_31.permute(0, 2, 1)
        view_31 = None
        permute_34 = x_250.permute(0, 2, 1)
        x_250 = None
        x_251 = permute_34.view(1, 320, 32, 32)
        permute_34 = None
        unsqueeze_64 = l_self_modules_backbone_modules_block3_modules_8_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_8_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_65 = unsqueeze_64.unsqueeze(-1)
        unsqueeze_64 = None
        batch_norm_36 = torch.nn.functional.batch_norm(
            x_251,
            l_self_modules_backbone_modules_block3_modules_8_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_8_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_8_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_8_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_8_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_norm1_parameters_bias_ = (None)
        shorcut_16 = batch_norm_36.clone()
        x_252 = torch.conv2d(
            batch_norm_36,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_36 = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_253 = torch._C._nn.gelu(x_252, approximate="none")
        x_252 = None
        u_16 = x_253.clone()
        attn_143 = torch.conv2d(
            x_253,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_253 = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_144 = torch.conv2d(
            attn_143,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_145 = torch.conv2d(
            attn_144,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_144 = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_146 = torch.conv2d(
            attn_143,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_147 = torch.conv2d(
            attn_146,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_146 = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_148 = torch.conv2d(
            attn_143,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_149 = torch.conv2d(
            attn_148,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_148 = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_96 = attn_143 + attn_145
        attn_143 = attn_145 = None
        add_97 = add_96 + attn_147
        add_96 = attn_147 = None
        attn_150 = add_97 + attn_149
        add_97 = attn_149 = None
        attn_151 = torch.conv2d(
            attn_150,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_150 = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_254 = attn_151 * u_16
        attn_151 = u_16 = None
        x_255 = torch.conv2d(
            x_254,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_254 = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_256 = x_255 + shorcut_16
        x_255 = shorcut_16 = None
        mul_49 = unsqueeze_65 * x_256
        unsqueeze_65 = x_256 = None
        x_257 = x_251 + mul_49
        x_251 = mul_49 = None
        unsqueeze_66 = l_self_modules_backbone_modules_block3_modules_8_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_8_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_67 = unsqueeze_66.unsqueeze(-1)
        unsqueeze_66 = None
        batch_norm_37 = torch.nn.functional.batch_norm(
            x_257,
            l_self_modules_backbone_modules_block3_modules_8_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_8_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_8_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_8_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_8_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_norm2_parameters_bias_ = (None)
        x_258 = torch.conv2d(
            batch_norm_37,
            l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_37 = l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_259 = torch.conv2d(
            x_258,
            l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_258 = l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_260 = torch._C._nn.gelu(x_259, approximate="none")
        x_259 = None
        x_261 = torch.nn.functional.dropout(x_260, 0.0, False, False)
        x_260 = None
        x_262 = torch.conv2d(
            x_261,
            l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_261 = l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_263 = torch.nn.functional.dropout(x_262, 0.0, False, False)
        x_262 = None
        mul_50 = unsqueeze_67 * x_263
        unsqueeze_67 = x_263 = None
        x_264 = x_257 + mul_50
        x_257 = mul_50 = None
        view_33 = x_264.view(1, 320, 1024)
        x_264 = None
        x_265 = view_33.permute(0, 2, 1)
        view_33 = None
        permute_36 = x_265.permute(0, 2, 1)
        x_265 = None
        x_266 = permute_36.view(1, 320, 32, 32)
        permute_36 = None
        unsqueeze_68 = l_self_modules_backbone_modules_block3_modules_9_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_9_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_69 = unsqueeze_68.unsqueeze(-1)
        unsqueeze_68 = None
        batch_norm_38 = torch.nn.functional.batch_norm(
            x_266,
            l_self_modules_backbone_modules_block3_modules_9_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_9_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_9_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_9_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_9_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_norm1_parameters_bias_ = (None)
        shorcut_17 = batch_norm_38.clone()
        x_267 = torch.conv2d(
            batch_norm_38,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_38 = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_268 = torch._C._nn.gelu(x_267, approximate="none")
        x_267 = None
        u_17 = x_268.clone()
        attn_152 = torch.conv2d(
            x_268,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_268 = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_153 = torch.conv2d(
            attn_152,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_154 = torch.conv2d(
            attn_153,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_153 = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_155 = torch.conv2d(
            attn_152,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_156 = torch.conv2d(
            attn_155,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_155 = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_157 = torch.conv2d(
            attn_152,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_158 = torch.conv2d(
            attn_157,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_157 = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_102 = attn_152 + attn_154
        attn_152 = attn_154 = None
        add_103 = add_102 + attn_156
        add_102 = attn_156 = None
        attn_159 = add_103 + attn_158
        add_103 = attn_158 = None
        attn_160 = torch.conv2d(
            attn_159,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_159 = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_269 = attn_160 * u_17
        attn_160 = u_17 = None
        x_270 = torch.conv2d(
            x_269,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_269 = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_271 = x_270 + shorcut_17
        x_270 = shorcut_17 = None
        mul_52 = unsqueeze_69 * x_271
        unsqueeze_69 = x_271 = None
        x_272 = x_266 + mul_52
        x_266 = mul_52 = None
        unsqueeze_70 = l_self_modules_backbone_modules_block3_modules_9_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_9_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_71 = unsqueeze_70.unsqueeze(-1)
        unsqueeze_70 = None
        batch_norm_39 = torch.nn.functional.batch_norm(
            x_272,
            l_self_modules_backbone_modules_block3_modules_9_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_9_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_9_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_9_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_9_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_norm2_parameters_bias_ = (None)
        x_273 = torch.conv2d(
            batch_norm_39,
            l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_39 = l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_273 = l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_275 = torch._C._nn.gelu(x_274, approximate="none")
        x_274 = None
        x_276 = torch.nn.functional.dropout(x_275, 0.0, False, False)
        x_275 = None
        x_277 = torch.conv2d(
            x_276,
            l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_276 = l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_278 = torch.nn.functional.dropout(x_277, 0.0, False, False)
        x_277 = None
        mul_53 = unsqueeze_71 * x_278
        unsqueeze_71 = x_278 = None
        x_279 = x_272 + mul_53
        x_272 = mul_53 = None
        view_35 = x_279.view(1, 320, 1024)
        x_279 = None
        x_280 = view_35.permute(0, 2, 1)
        view_35 = None
        permute_38 = x_280.permute(0, 2, 1)
        x_280 = None
        x_281 = permute_38.view(1, 320, 32, 32)
        permute_38 = None
        unsqueeze_72 = l_self_modules_backbone_modules_block3_modules_10_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_10_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_73 = unsqueeze_72.unsqueeze(-1)
        unsqueeze_72 = None
        batch_norm_40 = torch.nn.functional.batch_norm(
            x_281,
            l_self_modules_backbone_modules_block3_modules_10_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_10_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_10_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_10_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_10_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_norm1_parameters_bias_ = (None)
        shorcut_18 = batch_norm_40.clone()
        x_282 = torch.conv2d(
            batch_norm_40,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_40 = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_283 = torch._C._nn.gelu(x_282, approximate="none")
        x_282 = None
        u_18 = x_283.clone()
        attn_161 = torch.conv2d(
            x_283,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_283 = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_162 = torch.conv2d(
            attn_161,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_163 = torch.conv2d(
            attn_162,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_162 = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_164 = torch.conv2d(
            attn_161,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_165 = torch.conv2d(
            attn_164,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_164 = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_166 = torch.conv2d(
            attn_161,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_167 = torch.conv2d(
            attn_166,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_166 = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_108 = attn_161 + attn_163
        attn_161 = attn_163 = None
        add_109 = add_108 + attn_165
        add_108 = attn_165 = None
        attn_168 = add_109 + attn_167
        add_109 = attn_167 = None
        attn_169 = torch.conv2d(
            attn_168,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_168 = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_284 = attn_169 * u_18
        attn_169 = u_18 = None
        x_285 = torch.conv2d(
            x_284,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_284 = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_286 = x_285 + shorcut_18
        x_285 = shorcut_18 = None
        mul_55 = unsqueeze_73 * x_286
        unsqueeze_73 = x_286 = None
        x_287 = x_281 + mul_55
        x_281 = mul_55 = None
        unsqueeze_74 = l_self_modules_backbone_modules_block3_modules_10_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_10_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_75 = unsqueeze_74.unsqueeze(-1)
        unsqueeze_74 = None
        batch_norm_41 = torch.nn.functional.batch_norm(
            x_287,
            l_self_modules_backbone_modules_block3_modules_10_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_10_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_10_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_10_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_10_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_norm2_parameters_bias_ = (None)
        x_288 = torch.conv2d(
            batch_norm_41,
            l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_41 = l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_289 = torch.conv2d(
            x_288,
            l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_288 = l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_290 = torch._C._nn.gelu(x_289, approximate="none")
        x_289 = None
        x_291 = torch.nn.functional.dropout(x_290, 0.0, False, False)
        x_290 = None
        x_292 = torch.conv2d(
            x_291,
            l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_291 = l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_293 = torch.nn.functional.dropout(x_292, 0.0, False, False)
        x_292 = None
        mul_56 = unsqueeze_75 * x_293
        unsqueeze_75 = x_293 = None
        x_294 = x_287 + mul_56
        x_287 = mul_56 = None
        view_37 = x_294.view(1, 320, 1024)
        x_294 = None
        x_295 = view_37.permute(0, 2, 1)
        view_37 = None
        permute_40 = x_295.permute(0, 2, 1)
        x_295 = None
        x_296 = permute_40.view(1, 320, 32, 32)
        permute_40 = None
        unsqueeze_76 = l_self_modules_backbone_modules_block3_modules_11_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_11_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_77 = unsqueeze_76.unsqueeze(-1)
        unsqueeze_76 = None
        batch_norm_42 = torch.nn.functional.batch_norm(
            x_296,
            l_self_modules_backbone_modules_block3_modules_11_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_11_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_11_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_11_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_11_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_norm1_parameters_bias_ = (None)
        shorcut_19 = batch_norm_42.clone()
        x_297 = torch.conv2d(
            batch_norm_42,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_42 = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_298 = torch._C._nn.gelu(x_297, approximate="none")
        x_297 = None
        u_19 = x_298.clone()
        attn_170 = torch.conv2d(
            x_298,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_298 = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_171 = torch.conv2d(
            attn_170,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_172 = torch.conv2d(
            attn_171,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_171 = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_173 = torch.conv2d(
            attn_170,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_174 = torch.conv2d(
            attn_173,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_173 = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_175 = torch.conv2d(
            attn_170,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_176 = torch.conv2d(
            attn_175,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_175 = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_114 = attn_170 + attn_172
        attn_170 = attn_172 = None
        add_115 = add_114 + attn_174
        add_114 = attn_174 = None
        attn_177 = add_115 + attn_176
        add_115 = attn_176 = None
        attn_178 = torch.conv2d(
            attn_177,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_177 = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_299 = attn_178 * u_19
        attn_178 = u_19 = None
        x_300 = torch.conv2d(
            x_299,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_299 = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_301 = x_300 + shorcut_19
        x_300 = shorcut_19 = None
        mul_58 = unsqueeze_77 * x_301
        unsqueeze_77 = x_301 = None
        x_302 = x_296 + mul_58
        x_296 = mul_58 = None
        unsqueeze_78 = l_self_modules_backbone_modules_block3_modules_11_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_11_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_79 = unsqueeze_78.unsqueeze(-1)
        unsqueeze_78 = None
        batch_norm_43 = torch.nn.functional.batch_norm(
            x_302,
            l_self_modules_backbone_modules_block3_modules_11_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_11_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_11_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_11_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_11_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_norm2_parameters_bias_ = (None)
        x_303 = torch.conv2d(
            batch_norm_43,
            l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_43 = l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_304 = torch.conv2d(
            x_303,
            l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_303 = l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_305 = torch._C._nn.gelu(x_304, approximate="none")
        x_304 = None
        x_306 = torch.nn.functional.dropout(x_305, 0.0, False, False)
        x_305 = None
        x_307 = torch.conv2d(
            x_306,
            l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_306 = l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_308 = torch.nn.functional.dropout(x_307, 0.0, False, False)
        x_307 = None
        mul_59 = unsqueeze_79 * x_308
        unsqueeze_79 = x_308 = None
        x_309 = x_302 + mul_59
        x_302 = mul_59 = None
        view_39 = x_309.view(1, 320, 1024)
        x_309 = None
        x_310 = view_39.permute(0, 2, 1)
        view_39 = None
        permute_42 = x_310.permute(0, 2, 1)
        x_310 = None
        x_311 = permute_42.view(1, 320, 32, 32)
        permute_42 = None
        unsqueeze_80 = l_self_modules_backbone_modules_block3_modules_12_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_12_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_81 = unsqueeze_80.unsqueeze(-1)
        unsqueeze_80 = None
        batch_norm_44 = torch.nn.functional.batch_norm(
            x_311,
            l_self_modules_backbone_modules_block3_modules_12_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_12_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_12_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_12_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_12_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_norm1_parameters_bias_ = (None)
        shorcut_20 = batch_norm_44.clone()
        x_312 = torch.conv2d(
            batch_norm_44,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_44 = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_313 = torch._C._nn.gelu(x_312, approximate="none")
        x_312 = None
        u_20 = x_313.clone()
        attn_179 = torch.conv2d(
            x_313,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_313 = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_180 = torch.conv2d(
            attn_179,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_181 = torch.conv2d(
            attn_180,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_180 = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_182 = torch.conv2d(
            attn_179,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_183 = torch.conv2d(
            attn_182,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_182 = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_184 = torch.conv2d(
            attn_179,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_185 = torch.conv2d(
            attn_184,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_184 = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_120 = attn_179 + attn_181
        attn_179 = attn_181 = None
        add_121 = add_120 + attn_183
        add_120 = attn_183 = None
        attn_186 = add_121 + attn_185
        add_121 = attn_185 = None
        attn_187 = torch.conv2d(
            attn_186,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_186 = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_314 = attn_187 * u_20
        attn_187 = u_20 = None
        x_315 = torch.conv2d(
            x_314,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_314 = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_316 = x_315 + shorcut_20
        x_315 = shorcut_20 = None
        mul_61 = unsqueeze_81 * x_316
        unsqueeze_81 = x_316 = None
        x_317 = x_311 + mul_61
        x_311 = mul_61 = None
        unsqueeze_82 = l_self_modules_backbone_modules_block3_modules_12_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_12_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_83 = unsqueeze_82.unsqueeze(-1)
        unsqueeze_82 = None
        batch_norm_45 = torch.nn.functional.batch_norm(
            x_317,
            l_self_modules_backbone_modules_block3_modules_12_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_12_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_12_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_12_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_12_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_norm2_parameters_bias_ = (None)
        x_318 = torch.conv2d(
            batch_norm_45,
            l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_45 = l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_319 = torch.conv2d(
            x_318,
            l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_318 = l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_320 = torch._C._nn.gelu(x_319, approximate="none")
        x_319 = None
        x_321 = torch.nn.functional.dropout(x_320, 0.0, False, False)
        x_320 = None
        x_322 = torch.conv2d(
            x_321,
            l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_321 = l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_323 = torch.nn.functional.dropout(x_322, 0.0, False, False)
        x_322 = None
        mul_62 = unsqueeze_83 * x_323
        unsqueeze_83 = x_323 = None
        x_324 = x_317 + mul_62
        x_317 = mul_62 = None
        view_41 = x_324.view(1, 320, 1024)
        x_324 = None
        x_325 = view_41.permute(0, 2, 1)
        view_41 = None
        permute_44 = x_325.permute(0, 2, 1)
        x_325 = None
        x_326 = permute_44.view(1, 320, 32, 32)
        permute_44 = None
        unsqueeze_84 = l_self_modules_backbone_modules_block3_modules_13_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_13_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_85 = unsqueeze_84.unsqueeze(-1)
        unsqueeze_84 = None
        batch_norm_46 = torch.nn.functional.batch_norm(
            x_326,
            l_self_modules_backbone_modules_block3_modules_13_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_13_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_13_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_13_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_13_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_norm1_parameters_bias_ = (None)
        shorcut_21 = batch_norm_46.clone()
        x_327 = torch.conv2d(
            batch_norm_46,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_46 = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_328 = torch._C._nn.gelu(x_327, approximate="none")
        x_327 = None
        u_21 = x_328.clone()
        attn_188 = torch.conv2d(
            x_328,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_328 = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_189 = torch.conv2d(
            attn_188,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_190 = torch.conv2d(
            attn_189,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_189 = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_191 = torch.conv2d(
            attn_188,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_192 = torch.conv2d(
            attn_191,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_191 = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_193 = torch.conv2d(
            attn_188,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_194 = torch.conv2d(
            attn_193,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_193 = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_126 = attn_188 + attn_190
        attn_188 = attn_190 = None
        add_127 = add_126 + attn_192
        add_126 = attn_192 = None
        attn_195 = add_127 + attn_194
        add_127 = attn_194 = None
        attn_196 = torch.conv2d(
            attn_195,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_195 = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_329 = attn_196 * u_21
        attn_196 = u_21 = None
        x_330 = torch.conv2d(
            x_329,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_329 = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_331 = x_330 + shorcut_21
        x_330 = shorcut_21 = None
        mul_64 = unsqueeze_85 * x_331
        unsqueeze_85 = x_331 = None
        x_332 = x_326 + mul_64
        x_326 = mul_64 = None
        unsqueeze_86 = l_self_modules_backbone_modules_block3_modules_13_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_13_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_87 = unsqueeze_86.unsqueeze(-1)
        unsqueeze_86 = None
        batch_norm_47 = torch.nn.functional.batch_norm(
            x_332,
            l_self_modules_backbone_modules_block3_modules_13_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_13_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_13_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_13_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_13_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_norm2_parameters_bias_ = (None)
        x_333 = torch.conv2d(
            batch_norm_47,
            l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_47 = l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_334 = torch.conv2d(
            x_333,
            l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_333 = l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_335 = torch._C._nn.gelu(x_334, approximate="none")
        x_334 = None
        x_336 = torch.nn.functional.dropout(x_335, 0.0, False, False)
        x_335 = None
        x_337 = torch.conv2d(
            x_336,
            l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_336 = l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_338 = torch.nn.functional.dropout(x_337, 0.0, False, False)
        x_337 = None
        mul_65 = unsqueeze_87 * x_338
        unsqueeze_87 = x_338 = None
        x_339 = x_332 + mul_65
        x_332 = mul_65 = None
        view_43 = x_339.view(1, 320, 1024)
        x_339 = None
        x_340 = view_43.permute(0, 2, 1)
        view_43 = None
        permute_46 = x_340.permute(0, 2, 1)
        x_340 = None
        x_341 = permute_46.view(1, 320, 32, 32)
        permute_46 = None
        unsqueeze_88 = l_self_modules_backbone_modules_block3_modules_14_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_14_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_89 = unsqueeze_88.unsqueeze(-1)
        unsqueeze_88 = None
        batch_norm_48 = torch.nn.functional.batch_norm(
            x_341,
            l_self_modules_backbone_modules_block3_modules_14_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_14_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_14_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_14_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_14_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_norm1_parameters_bias_ = (None)
        shorcut_22 = batch_norm_48.clone()
        x_342 = torch.conv2d(
            batch_norm_48,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_48 = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_343 = torch._C._nn.gelu(x_342, approximate="none")
        x_342 = None
        u_22 = x_343.clone()
        attn_197 = torch.conv2d(
            x_343,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_343 = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_198 = torch.conv2d(
            attn_197,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_199 = torch.conv2d(
            attn_198,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_198 = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_200 = torch.conv2d(
            attn_197,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_201 = torch.conv2d(
            attn_200,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_200 = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_202 = torch.conv2d(
            attn_197,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_203 = torch.conv2d(
            attn_202,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_202 = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_132 = attn_197 + attn_199
        attn_197 = attn_199 = None
        add_133 = add_132 + attn_201
        add_132 = attn_201 = None
        attn_204 = add_133 + attn_203
        add_133 = attn_203 = None
        attn_205 = torch.conv2d(
            attn_204,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_204 = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_344 = attn_205 * u_22
        attn_205 = u_22 = None
        x_345 = torch.conv2d(
            x_344,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_344 = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_346 = x_345 + shorcut_22
        x_345 = shorcut_22 = None
        mul_67 = unsqueeze_89 * x_346
        unsqueeze_89 = x_346 = None
        x_347 = x_341 + mul_67
        x_341 = mul_67 = None
        unsqueeze_90 = l_self_modules_backbone_modules_block3_modules_14_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_14_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_91 = unsqueeze_90.unsqueeze(-1)
        unsqueeze_90 = None
        batch_norm_49 = torch.nn.functional.batch_norm(
            x_347,
            l_self_modules_backbone_modules_block3_modules_14_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_14_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_14_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_14_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_14_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_norm2_parameters_bias_ = (None)
        x_348 = torch.conv2d(
            batch_norm_49,
            l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_49 = l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_349 = torch.conv2d(
            x_348,
            l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_348 = l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_350 = torch._C._nn.gelu(x_349, approximate="none")
        x_349 = None
        x_351 = torch.nn.functional.dropout(x_350, 0.0, False, False)
        x_350 = None
        x_352 = torch.conv2d(
            x_351,
            l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_351 = l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_353 = torch.nn.functional.dropout(x_352, 0.0, False, False)
        x_352 = None
        mul_68 = unsqueeze_91 * x_353
        unsqueeze_91 = x_353 = None
        x_354 = x_347 + mul_68
        x_347 = mul_68 = None
        view_45 = x_354.view(1, 320, 1024)
        x_354 = None
        x_355 = view_45.permute(0, 2, 1)
        view_45 = None
        permute_48 = x_355.permute(0, 2, 1)
        x_355 = None
        x_356 = permute_48.view(1, 320, 32, 32)
        permute_48 = None
        unsqueeze_92 = l_self_modules_backbone_modules_block3_modules_15_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_15_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_93 = unsqueeze_92.unsqueeze(-1)
        unsqueeze_92 = None
        batch_norm_50 = torch.nn.functional.batch_norm(
            x_356,
            l_self_modules_backbone_modules_block3_modules_15_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_15_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_15_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_15_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_15_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_norm1_parameters_bias_ = (None)
        shorcut_23 = batch_norm_50.clone()
        x_357 = torch.conv2d(
            batch_norm_50,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_50 = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_358 = torch._C._nn.gelu(x_357, approximate="none")
        x_357 = None
        u_23 = x_358.clone()
        attn_206 = torch.conv2d(
            x_358,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_358 = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_207 = torch.conv2d(
            attn_206,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_208 = torch.conv2d(
            attn_207,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_207 = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_209 = torch.conv2d(
            attn_206,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_210 = torch.conv2d(
            attn_209,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_209 = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_211 = torch.conv2d(
            attn_206,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_212 = torch.conv2d(
            attn_211,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_211 = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_138 = attn_206 + attn_208
        attn_206 = attn_208 = None
        add_139 = add_138 + attn_210
        add_138 = attn_210 = None
        attn_213 = add_139 + attn_212
        add_139 = attn_212 = None
        attn_214 = torch.conv2d(
            attn_213,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_213 = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_359 = attn_214 * u_23
        attn_214 = u_23 = None
        x_360 = torch.conv2d(
            x_359,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_359 = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_361 = x_360 + shorcut_23
        x_360 = shorcut_23 = None
        mul_70 = unsqueeze_93 * x_361
        unsqueeze_93 = x_361 = None
        x_362 = x_356 + mul_70
        x_356 = mul_70 = None
        unsqueeze_94 = l_self_modules_backbone_modules_block3_modules_15_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_15_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_95 = unsqueeze_94.unsqueeze(-1)
        unsqueeze_94 = None
        batch_norm_51 = torch.nn.functional.batch_norm(
            x_362,
            l_self_modules_backbone_modules_block3_modules_15_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_15_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_15_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_15_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_15_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_norm2_parameters_bias_ = (None)
        x_363 = torch.conv2d(
            batch_norm_51,
            l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_51 = l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_364 = torch.conv2d(
            x_363,
            l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_363 = l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_365 = torch._C._nn.gelu(x_364, approximate="none")
        x_364 = None
        x_366 = torch.nn.functional.dropout(x_365, 0.0, False, False)
        x_365 = None
        x_367 = torch.conv2d(
            x_366,
            l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_366 = l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_368 = torch.nn.functional.dropout(x_367, 0.0, False, False)
        x_367 = None
        mul_71 = unsqueeze_95 * x_368
        unsqueeze_95 = x_368 = None
        x_369 = x_362 + mul_71
        x_362 = mul_71 = None
        view_47 = x_369.view(1, 320, 1024)
        x_369 = None
        x_370 = view_47.permute(0, 2, 1)
        view_47 = None
        permute_50 = x_370.permute(0, 2, 1)
        x_370 = None
        x_371 = permute_50.view(1, 320, 32, 32)
        permute_50 = None
        unsqueeze_96 = l_self_modules_backbone_modules_block3_modules_16_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_16_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_97 = unsqueeze_96.unsqueeze(-1)
        unsqueeze_96 = None
        batch_norm_52 = torch.nn.functional.batch_norm(
            x_371,
            l_self_modules_backbone_modules_block3_modules_16_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_16_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_16_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_16_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_16_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_norm1_parameters_bias_ = (None)
        shorcut_24 = batch_norm_52.clone()
        x_372 = torch.conv2d(
            batch_norm_52,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_52 = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_373 = torch._C._nn.gelu(x_372, approximate="none")
        x_372 = None
        u_24 = x_373.clone()
        attn_215 = torch.conv2d(
            x_373,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_373 = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_216 = torch.conv2d(
            attn_215,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_217 = torch.conv2d(
            attn_216,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_216 = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_218 = torch.conv2d(
            attn_215,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_219 = torch.conv2d(
            attn_218,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_218 = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_220 = torch.conv2d(
            attn_215,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_221 = torch.conv2d(
            attn_220,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_220 = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_144 = attn_215 + attn_217
        attn_215 = attn_217 = None
        add_145 = add_144 + attn_219
        add_144 = attn_219 = None
        attn_222 = add_145 + attn_221
        add_145 = attn_221 = None
        attn_223 = torch.conv2d(
            attn_222,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_222 = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_374 = attn_223 * u_24
        attn_223 = u_24 = None
        x_375 = torch.conv2d(
            x_374,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_374 = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_376 = x_375 + shorcut_24
        x_375 = shorcut_24 = None
        mul_73 = unsqueeze_97 * x_376
        unsqueeze_97 = x_376 = None
        x_377 = x_371 + mul_73
        x_371 = mul_73 = None
        unsqueeze_98 = l_self_modules_backbone_modules_block3_modules_16_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_16_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_99 = unsqueeze_98.unsqueeze(-1)
        unsqueeze_98 = None
        batch_norm_53 = torch.nn.functional.batch_norm(
            x_377,
            l_self_modules_backbone_modules_block3_modules_16_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_16_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_16_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_16_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_16_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_norm2_parameters_bias_ = (None)
        x_378 = torch.conv2d(
            batch_norm_53,
            l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_53 = l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_379 = torch.conv2d(
            x_378,
            l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_378 = l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_380 = torch._C._nn.gelu(x_379, approximate="none")
        x_379 = None
        x_381 = torch.nn.functional.dropout(x_380, 0.0, False, False)
        x_380 = None
        x_382 = torch.conv2d(
            x_381,
            l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_381 = l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_383 = torch.nn.functional.dropout(x_382, 0.0, False, False)
        x_382 = None
        mul_74 = unsqueeze_99 * x_383
        unsqueeze_99 = x_383 = None
        x_384 = x_377 + mul_74
        x_377 = mul_74 = None
        view_49 = x_384.view(1, 320, 1024)
        x_384 = None
        x_385 = view_49.permute(0, 2, 1)
        view_49 = None
        permute_52 = x_385.permute(0, 2, 1)
        x_385 = None
        x_386 = permute_52.view(1, 320, 32, 32)
        permute_52 = None
        unsqueeze_100 = l_self_modules_backbone_modules_block3_modules_17_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_17_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_101 = unsqueeze_100.unsqueeze(-1)
        unsqueeze_100 = None
        batch_norm_54 = torch.nn.functional.batch_norm(
            x_386,
            l_self_modules_backbone_modules_block3_modules_17_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_17_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_17_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_17_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_17_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_norm1_parameters_bias_ = (None)
        shorcut_25 = batch_norm_54.clone()
        x_387 = torch.conv2d(
            batch_norm_54,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_54 = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_388 = torch._C._nn.gelu(x_387, approximate="none")
        x_387 = None
        u_25 = x_388.clone()
        attn_224 = torch.conv2d(
            x_388,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_388 = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_225 = torch.conv2d(
            attn_224,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_226 = torch.conv2d(
            attn_225,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_225 = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_227 = torch.conv2d(
            attn_224,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_228 = torch.conv2d(
            attn_227,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_227 = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_229 = torch.conv2d(
            attn_224,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_230 = torch.conv2d(
            attn_229,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_229 = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_150 = attn_224 + attn_226
        attn_224 = attn_226 = None
        add_151 = add_150 + attn_228
        add_150 = attn_228 = None
        attn_231 = add_151 + attn_230
        add_151 = attn_230 = None
        attn_232 = torch.conv2d(
            attn_231,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_231 = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_389 = attn_232 * u_25
        attn_232 = u_25 = None
        x_390 = torch.conv2d(
            x_389,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_389 = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_391 = x_390 + shorcut_25
        x_390 = shorcut_25 = None
        mul_76 = unsqueeze_101 * x_391
        unsqueeze_101 = x_391 = None
        x_392 = x_386 + mul_76
        x_386 = mul_76 = None
        unsqueeze_102 = l_self_modules_backbone_modules_block3_modules_17_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_17_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_103 = unsqueeze_102.unsqueeze(-1)
        unsqueeze_102 = None
        batch_norm_55 = torch.nn.functional.batch_norm(
            x_392,
            l_self_modules_backbone_modules_block3_modules_17_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_17_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_17_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_17_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_17_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_norm2_parameters_bias_ = (None)
        x_393 = torch.conv2d(
            batch_norm_55,
            l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_55 = l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_394 = torch.conv2d(
            x_393,
            l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_393 = l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_395 = torch._C._nn.gelu(x_394, approximate="none")
        x_394 = None
        x_396 = torch.nn.functional.dropout(x_395, 0.0, False, False)
        x_395 = None
        x_397 = torch.conv2d(
            x_396,
            l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_396 = l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_398 = torch.nn.functional.dropout(x_397, 0.0, False, False)
        x_397 = None
        mul_77 = unsqueeze_103 * x_398
        unsqueeze_103 = x_398 = None
        x_399 = x_392 + mul_77
        x_392 = mul_77 = None
        view_51 = x_399.view(1, 320, 1024)
        x_399 = None
        x_400 = view_51.permute(0, 2, 1)
        view_51 = None
        permute_54 = x_400.permute(0, 2, 1)
        x_400 = None
        x_401 = permute_54.view(1, 320, 32, 32)
        permute_54 = None
        unsqueeze_104 = l_self_modules_backbone_modules_block3_modules_18_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_18_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_105 = unsqueeze_104.unsqueeze(-1)
        unsqueeze_104 = None
        batch_norm_56 = torch.nn.functional.batch_norm(
            x_401,
            l_self_modules_backbone_modules_block3_modules_18_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_18_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_18_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_18_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_18_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_norm1_parameters_bias_ = (None)
        shorcut_26 = batch_norm_56.clone()
        x_402 = torch.conv2d(
            batch_norm_56,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_56 = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_403 = torch._C._nn.gelu(x_402, approximate="none")
        x_402 = None
        u_26 = x_403.clone()
        attn_233 = torch.conv2d(
            x_403,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_403 = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_234 = torch.conv2d(
            attn_233,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_235 = torch.conv2d(
            attn_234,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_234 = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_236 = torch.conv2d(
            attn_233,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_237 = torch.conv2d(
            attn_236,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_236 = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_238 = torch.conv2d(
            attn_233,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_239 = torch.conv2d(
            attn_238,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_238 = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_156 = attn_233 + attn_235
        attn_233 = attn_235 = None
        add_157 = add_156 + attn_237
        add_156 = attn_237 = None
        attn_240 = add_157 + attn_239
        add_157 = attn_239 = None
        attn_241 = torch.conv2d(
            attn_240,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_240 = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_404 = attn_241 * u_26
        attn_241 = u_26 = None
        x_405 = torch.conv2d(
            x_404,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_404 = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_406 = x_405 + shorcut_26
        x_405 = shorcut_26 = None
        mul_79 = unsqueeze_105 * x_406
        unsqueeze_105 = x_406 = None
        x_407 = x_401 + mul_79
        x_401 = mul_79 = None
        unsqueeze_106 = l_self_modules_backbone_modules_block3_modules_18_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_18_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_107 = unsqueeze_106.unsqueeze(-1)
        unsqueeze_106 = None
        batch_norm_57 = torch.nn.functional.batch_norm(
            x_407,
            l_self_modules_backbone_modules_block3_modules_18_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_18_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_18_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_18_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_18_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_norm2_parameters_bias_ = (None)
        x_408 = torch.conv2d(
            batch_norm_57,
            l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_57 = l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_409 = torch.conv2d(
            x_408,
            l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_408 = l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_410 = torch._C._nn.gelu(x_409, approximate="none")
        x_409 = None
        x_411 = torch.nn.functional.dropout(x_410, 0.0, False, False)
        x_410 = None
        x_412 = torch.conv2d(
            x_411,
            l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_411 = l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_18_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_413 = torch.nn.functional.dropout(x_412, 0.0, False, False)
        x_412 = None
        mul_80 = unsqueeze_107 * x_413
        unsqueeze_107 = x_413 = None
        x_414 = x_407 + mul_80
        x_407 = mul_80 = None
        view_53 = x_414.view(1, 320, 1024)
        x_414 = None
        x_415 = view_53.permute(0, 2, 1)
        view_53 = None
        permute_56 = x_415.permute(0, 2, 1)
        x_415 = None
        x_416 = permute_56.view(1, 320, 32, 32)
        permute_56 = None
        unsqueeze_108 = l_self_modules_backbone_modules_block3_modules_19_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_19_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_109 = unsqueeze_108.unsqueeze(-1)
        unsqueeze_108 = None
        batch_norm_58 = torch.nn.functional.batch_norm(
            x_416,
            l_self_modules_backbone_modules_block3_modules_19_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_19_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_19_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_19_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_19_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_norm1_parameters_bias_ = (None)
        shorcut_27 = batch_norm_58.clone()
        x_417 = torch.conv2d(
            batch_norm_58,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_58 = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_418 = torch._C._nn.gelu(x_417, approximate="none")
        x_417 = None
        u_27 = x_418.clone()
        attn_242 = torch.conv2d(
            x_418,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_418 = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_243 = torch.conv2d(
            attn_242,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_244 = torch.conv2d(
            attn_243,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_243 = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_245 = torch.conv2d(
            attn_242,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_246 = torch.conv2d(
            attn_245,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_245 = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_247 = torch.conv2d(
            attn_242,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_248 = torch.conv2d(
            attn_247,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_247 = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_162 = attn_242 + attn_244
        attn_242 = attn_244 = None
        add_163 = add_162 + attn_246
        add_162 = attn_246 = None
        attn_249 = add_163 + attn_248
        add_163 = attn_248 = None
        attn_250 = torch.conv2d(
            attn_249,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_249 = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_419 = attn_250 * u_27
        attn_250 = u_27 = None
        x_420 = torch.conv2d(
            x_419,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_419 = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_421 = x_420 + shorcut_27
        x_420 = shorcut_27 = None
        mul_82 = unsqueeze_109 * x_421
        unsqueeze_109 = x_421 = None
        x_422 = x_416 + mul_82
        x_416 = mul_82 = None
        unsqueeze_110 = l_self_modules_backbone_modules_block3_modules_19_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_19_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_111 = unsqueeze_110.unsqueeze(-1)
        unsqueeze_110 = None
        batch_norm_59 = torch.nn.functional.batch_norm(
            x_422,
            l_self_modules_backbone_modules_block3_modules_19_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_19_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_19_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_19_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_19_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_norm2_parameters_bias_ = (None)
        x_423 = torch.conv2d(
            batch_norm_59,
            l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_59 = l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_424 = torch.conv2d(
            x_423,
            l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_423 = l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_425 = torch._C._nn.gelu(x_424, approximate="none")
        x_424 = None
        x_426 = torch.nn.functional.dropout(x_425, 0.0, False, False)
        x_425 = None
        x_427 = torch.conv2d(
            x_426,
            l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_426 = l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_19_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_428 = torch.nn.functional.dropout(x_427, 0.0, False, False)
        x_427 = None
        mul_83 = unsqueeze_111 * x_428
        unsqueeze_111 = x_428 = None
        x_429 = x_422 + mul_83
        x_422 = mul_83 = None
        view_55 = x_429.view(1, 320, 1024)
        x_429 = None
        x_430 = view_55.permute(0, 2, 1)
        view_55 = None
        permute_58 = x_430.permute(0, 2, 1)
        x_430 = None
        x_431 = permute_58.view(1, 320, 32, 32)
        permute_58 = None
        unsqueeze_112 = l_self_modules_backbone_modules_block3_modules_20_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_20_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_113 = unsqueeze_112.unsqueeze(-1)
        unsqueeze_112 = None
        batch_norm_60 = torch.nn.functional.batch_norm(
            x_431,
            l_self_modules_backbone_modules_block3_modules_20_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_20_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_20_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_20_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_20_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_norm1_parameters_bias_ = (None)
        shorcut_28 = batch_norm_60.clone()
        x_432 = torch.conv2d(
            batch_norm_60,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_60 = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_433 = torch._C._nn.gelu(x_432, approximate="none")
        x_432 = None
        u_28 = x_433.clone()
        attn_251 = torch.conv2d(
            x_433,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_433 = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_252 = torch.conv2d(
            attn_251,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_253 = torch.conv2d(
            attn_252,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_252 = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_254 = torch.conv2d(
            attn_251,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_255 = torch.conv2d(
            attn_254,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_254 = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_256 = torch.conv2d(
            attn_251,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_257 = torch.conv2d(
            attn_256,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_256 = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_168 = attn_251 + attn_253
        attn_251 = attn_253 = None
        add_169 = add_168 + attn_255
        add_168 = attn_255 = None
        attn_258 = add_169 + attn_257
        add_169 = attn_257 = None
        attn_259 = torch.conv2d(
            attn_258,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_258 = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_434 = attn_259 * u_28
        attn_259 = u_28 = None
        x_435 = torch.conv2d(
            x_434,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_434 = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_436 = x_435 + shorcut_28
        x_435 = shorcut_28 = None
        mul_85 = unsqueeze_113 * x_436
        unsqueeze_113 = x_436 = None
        x_437 = x_431 + mul_85
        x_431 = mul_85 = None
        unsqueeze_114 = l_self_modules_backbone_modules_block3_modules_20_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_20_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_115 = unsqueeze_114.unsqueeze(-1)
        unsqueeze_114 = None
        batch_norm_61 = torch.nn.functional.batch_norm(
            x_437,
            l_self_modules_backbone_modules_block3_modules_20_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_20_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_20_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_20_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_20_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_norm2_parameters_bias_ = (None)
        x_438 = torch.conv2d(
            batch_norm_61,
            l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_61 = l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_439 = torch.conv2d(
            x_438,
            l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_438 = l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_440 = torch._C._nn.gelu(x_439, approximate="none")
        x_439 = None
        x_441 = torch.nn.functional.dropout(x_440, 0.0, False, False)
        x_440 = None
        x_442 = torch.conv2d(
            x_441,
            l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_441 = l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_20_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_443 = torch.nn.functional.dropout(x_442, 0.0, False, False)
        x_442 = None
        mul_86 = unsqueeze_115 * x_443
        unsqueeze_115 = x_443 = None
        x_444 = x_437 + mul_86
        x_437 = mul_86 = None
        view_57 = x_444.view(1, 320, 1024)
        x_444 = None
        x_445 = view_57.permute(0, 2, 1)
        view_57 = None
        permute_60 = x_445.permute(0, 2, 1)
        x_445 = None
        x_446 = permute_60.view(1, 320, 32, 32)
        permute_60 = None
        unsqueeze_116 = l_self_modules_backbone_modules_block3_modules_21_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_21_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_117 = unsqueeze_116.unsqueeze(-1)
        unsqueeze_116 = None
        batch_norm_62 = torch.nn.functional.batch_norm(
            x_446,
            l_self_modules_backbone_modules_block3_modules_21_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_21_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_21_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_21_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_21_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_norm1_parameters_bias_ = (None)
        shorcut_29 = batch_norm_62.clone()
        x_447 = torch.conv2d(
            batch_norm_62,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_62 = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_448 = torch._C._nn.gelu(x_447, approximate="none")
        x_447 = None
        u_29 = x_448.clone()
        attn_260 = torch.conv2d(
            x_448,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_448 = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_261 = torch.conv2d(
            attn_260,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_262 = torch.conv2d(
            attn_261,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_261 = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_263 = torch.conv2d(
            attn_260,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_264 = torch.conv2d(
            attn_263,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_263 = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_265 = torch.conv2d(
            attn_260,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_266 = torch.conv2d(
            attn_265,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_265 = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_174 = attn_260 + attn_262
        attn_260 = attn_262 = None
        add_175 = add_174 + attn_264
        add_174 = attn_264 = None
        attn_267 = add_175 + attn_266
        add_175 = attn_266 = None
        attn_268 = torch.conv2d(
            attn_267,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_267 = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_449 = attn_268 * u_29
        attn_268 = u_29 = None
        x_450 = torch.conv2d(
            x_449,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_449 = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_451 = x_450 + shorcut_29
        x_450 = shorcut_29 = None
        mul_88 = unsqueeze_117 * x_451
        unsqueeze_117 = x_451 = None
        x_452 = x_446 + mul_88
        x_446 = mul_88 = None
        unsqueeze_118 = l_self_modules_backbone_modules_block3_modules_21_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_21_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_119 = unsqueeze_118.unsqueeze(-1)
        unsqueeze_118 = None
        batch_norm_63 = torch.nn.functional.batch_norm(
            x_452,
            l_self_modules_backbone_modules_block3_modules_21_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_21_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_21_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_21_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_21_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_norm2_parameters_bias_ = (None)
        x_453 = torch.conv2d(
            batch_norm_63,
            l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_63 = l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_454 = torch.conv2d(
            x_453,
            l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_453 = l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_455 = torch._C._nn.gelu(x_454, approximate="none")
        x_454 = None
        x_456 = torch.nn.functional.dropout(x_455, 0.0, False, False)
        x_455 = None
        x_457 = torch.conv2d(
            x_456,
            l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_456 = l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_21_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_458 = torch.nn.functional.dropout(x_457, 0.0, False, False)
        x_457 = None
        mul_89 = unsqueeze_119 * x_458
        unsqueeze_119 = x_458 = None
        x_459 = x_452 + mul_89
        x_452 = mul_89 = None
        view_59 = x_459.view(1, 320, 1024)
        x_459 = None
        x_460 = view_59.permute(0, 2, 1)
        view_59 = None
        permute_62 = x_460.permute(0, 2, 1)
        x_460 = None
        x_461 = permute_62.view(1, 320, 32, 32)
        permute_62 = None
        unsqueeze_120 = l_self_modules_backbone_modules_block3_modules_22_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_22_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_121 = unsqueeze_120.unsqueeze(-1)
        unsqueeze_120 = None
        batch_norm_64 = torch.nn.functional.batch_norm(
            x_461,
            l_self_modules_backbone_modules_block3_modules_22_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_22_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_22_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_22_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_22_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_norm1_parameters_bias_ = (None)
        shorcut_30 = batch_norm_64.clone()
        x_462 = torch.conv2d(
            batch_norm_64,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_64 = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_463 = torch._C._nn.gelu(x_462, approximate="none")
        x_462 = None
        u_30 = x_463.clone()
        attn_269 = torch.conv2d(
            x_463,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_463 = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_270 = torch.conv2d(
            attn_269,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_271 = torch.conv2d(
            attn_270,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_270 = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_272 = torch.conv2d(
            attn_269,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_273 = torch.conv2d(
            attn_272,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_272 = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_274 = torch.conv2d(
            attn_269,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_275 = torch.conv2d(
            attn_274,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_274 = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_180 = attn_269 + attn_271
        attn_269 = attn_271 = None
        add_181 = add_180 + attn_273
        add_180 = attn_273 = None
        attn_276 = add_181 + attn_275
        add_181 = attn_275 = None
        attn_277 = torch.conv2d(
            attn_276,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_276 = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_464 = attn_277 * u_30
        attn_277 = u_30 = None
        x_465 = torch.conv2d(
            x_464,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_464 = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_466 = x_465 + shorcut_30
        x_465 = shorcut_30 = None
        mul_91 = unsqueeze_121 * x_466
        unsqueeze_121 = x_466 = None
        x_467 = x_461 + mul_91
        x_461 = mul_91 = None
        unsqueeze_122 = l_self_modules_backbone_modules_block3_modules_22_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_22_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_123 = unsqueeze_122.unsqueeze(-1)
        unsqueeze_122 = None
        batch_norm_65 = torch.nn.functional.batch_norm(
            x_467,
            l_self_modules_backbone_modules_block3_modules_22_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_22_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_22_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_22_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_22_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_norm2_parameters_bias_ = (None)
        x_468 = torch.conv2d(
            batch_norm_65,
            l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_65 = l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_469 = torch.conv2d(
            x_468,
            l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_468 = l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_470 = torch._C._nn.gelu(x_469, approximate="none")
        x_469 = None
        x_471 = torch.nn.functional.dropout(x_470, 0.0, False, False)
        x_470 = None
        x_472 = torch.conv2d(
            x_471,
            l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_471 = l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_22_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_473 = torch.nn.functional.dropout(x_472, 0.0, False, False)
        x_472 = None
        mul_92 = unsqueeze_123 * x_473
        unsqueeze_123 = x_473 = None
        x_474 = x_467 + mul_92
        x_467 = mul_92 = None
        view_61 = x_474.view(1, 320, 1024)
        x_474 = None
        x_475 = view_61.permute(0, 2, 1)
        view_61 = None
        permute_64 = x_475.permute(0, 2, 1)
        x_475 = None
        x_476 = permute_64.view(1, 320, 32, 32)
        permute_64 = None
        unsqueeze_124 = l_self_modules_backbone_modules_block3_modules_23_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_23_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_125 = unsqueeze_124.unsqueeze(-1)
        unsqueeze_124 = None
        batch_norm_66 = torch.nn.functional.batch_norm(
            x_476,
            l_self_modules_backbone_modules_block3_modules_23_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_23_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_23_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_23_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_23_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_norm1_parameters_bias_ = (None)
        shorcut_31 = batch_norm_66.clone()
        x_477 = torch.conv2d(
            batch_norm_66,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_66 = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_478 = torch._C._nn.gelu(x_477, approximate="none")
        x_477 = None
        u_31 = x_478.clone()
        attn_278 = torch.conv2d(
            x_478,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_478 = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_279 = torch.conv2d(
            attn_278,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_280 = torch.conv2d(
            attn_279,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_279 = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_281 = torch.conv2d(
            attn_278,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_282 = torch.conv2d(
            attn_281,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_281 = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_283 = torch.conv2d(
            attn_278,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_284 = torch.conv2d(
            attn_283,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_283 = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_186 = attn_278 + attn_280
        attn_278 = attn_280 = None
        add_187 = add_186 + attn_282
        add_186 = attn_282 = None
        attn_285 = add_187 + attn_284
        add_187 = attn_284 = None
        attn_286 = torch.conv2d(
            attn_285,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_285 = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_479 = attn_286 * u_31
        attn_286 = u_31 = None
        x_480 = torch.conv2d(
            x_479,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_479 = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_481 = x_480 + shorcut_31
        x_480 = shorcut_31 = None
        mul_94 = unsqueeze_125 * x_481
        unsqueeze_125 = x_481 = None
        x_482 = x_476 + mul_94
        x_476 = mul_94 = None
        unsqueeze_126 = l_self_modules_backbone_modules_block3_modules_23_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_23_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_127 = unsqueeze_126.unsqueeze(-1)
        unsqueeze_126 = None
        batch_norm_67 = torch.nn.functional.batch_norm(
            x_482,
            l_self_modules_backbone_modules_block3_modules_23_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_23_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_23_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_23_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_23_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_norm2_parameters_bias_ = (None)
        x_483 = torch.conv2d(
            batch_norm_67,
            l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_67 = l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_484 = torch.conv2d(
            x_483,
            l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_483 = l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_485 = torch._C._nn.gelu(x_484, approximate="none")
        x_484 = None
        x_486 = torch.nn.functional.dropout(x_485, 0.0, False, False)
        x_485 = None
        x_487 = torch.conv2d(
            x_486,
            l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_486 = l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_488 = torch.nn.functional.dropout(x_487, 0.0, False, False)
        x_487 = None
        mul_95 = unsqueeze_127 * x_488
        unsqueeze_127 = x_488 = None
        x_489 = x_482 + mul_95
        x_482 = mul_95 = None
        view_63 = x_489.view(1, 320, 1024)
        x_489 = None
        x_490 = view_63.permute(0, 2, 1)
        view_63 = None
        permute_66 = x_490.permute(0, 2, 1)
        x_490 = None
        x_491 = permute_66.view(1, 320, 32, 32)
        permute_66 = None
        unsqueeze_128 = l_self_modules_backbone_modules_block3_modules_24_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_24_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_129 = unsqueeze_128.unsqueeze(-1)
        unsqueeze_128 = None
        batch_norm_68 = torch.nn.functional.batch_norm(
            x_491,
            l_self_modules_backbone_modules_block3_modules_24_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_24_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_24_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_24_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_24_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_24_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_norm1_parameters_bias_ = (None)
        shorcut_32 = batch_norm_68.clone()
        x_492 = torch.conv2d(
            batch_norm_68,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_68 = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_493 = torch._C._nn.gelu(x_492, approximate="none")
        x_492 = None
        u_32 = x_493.clone()
        attn_287 = torch.conv2d(
            x_493,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_493 = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_288 = torch.conv2d(
            attn_287,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_289 = torch.conv2d(
            attn_288,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_288 = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_290 = torch.conv2d(
            attn_287,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_291 = torch.conv2d(
            attn_290,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_290 = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_292 = torch.conv2d(
            attn_287,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_293 = torch.conv2d(
            attn_292,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_292 = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_192 = attn_287 + attn_289
        attn_287 = attn_289 = None
        add_193 = add_192 + attn_291
        add_192 = attn_291 = None
        attn_294 = add_193 + attn_293
        add_193 = attn_293 = None
        attn_295 = torch.conv2d(
            attn_294,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_294 = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_494 = attn_295 * u_32
        attn_295 = u_32 = None
        x_495 = torch.conv2d(
            x_494,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_494 = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_496 = x_495 + shorcut_32
        x_495 = shorcut_32 = None
        mul_97 = unsqueeze_129 * x_496
        unsqueeze_129 = x_496 = None
        x_497 = x_491 + mul_97
        x_491 = mul_97 = None
        unsqueeze_130 = l_self_modules_backbone_modules_block3_modules_24_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_24_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_131 = unsqueeze_130.unsqueeze(-1)
        unsqueeze_130 = None
        batch_norm_69 = torch.nn.functional.batch_norm(
            x_497,
            l_self_modules_backbone_modules_block3_modules_24_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_24_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_24_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_24_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_24_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_24_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_norm2_parameters_bias_ = (None)
        x_498 = torch.conv2d(
            batch_norm_69,
            l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_69 = l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_499 = torch.conv2d(
            x_498,
            l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_498 = l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_500 = torch._C._nn.gelu(x_499, approximate="none")
        x_499 = None
        x_501 = torch.nn.functional.dropout(x_500, 0.0, False, False)
        x_500 = None
        x_502 = torch.conv2d(
            x_501,
            l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_501 = l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_24_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_503 = torch.nn.functional.dropout(x_502, 0.0, False, False)
        x_502 = None
        mul_98 = unsqueeze_131 * x_503
        unsqueeze_131 = x_503 = None
        x_504 = x_497 + mul_98
        x_497 = mul_98 = None
        view_65 = x_504.view(1, 320, 1024)
        x_504 = None
        x_505 = view_65.permute(0, 2, 1)
        view_65 = None
        permute_68 = x_505.permute(0, 2, 1)
        x_505 = None
        x_506 = permute_68.view(1, 320, 32, 32)
        permute_68 = None
        unsqueeze_132 = l_self_modules_backbone_modules_block3_modules_25_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_25_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_133 = unsqueeze_132.unsqueeze(-1)
        unsqueeze_132 = None
        batch_norm_70 = torch.nn.functional.batch_norm(
            x_506,
            l_self_modules_backbone_modules_block3_modules_25_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_25_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_25_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_25_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_25_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_25_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_norm1_parameters_bias_ = (None)
        shorcut_33 = batch_norm_70.clone()
        x_507 = torch.conv2d(
            batch_norm_70,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_70 = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_508 = torch._C._nn.gelu(x_507, approximate="none")
        x_507 = None
        u_33 = x_508.clone()
        attn_296 = torch.conv2d(
            x_508,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_508 = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_297 = torch.conv2d(
            attn_296,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_298 = torch.conv2d(
            attn_297,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_297 = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_299 = torch.conv2d(
            attn_296,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_300 = torch.conv2d(
            attn_299,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_299 = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_301 = torch.conv2d(
            attn_296,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_302 = torch.conv2d(
            attn_301,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_301 = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_198 = attn_296 + attn_298
        attn_296 = attn_298 = None
        add_199 = add_198 + attn_300
        add_198 = attn_300 = None
        attn_303 = add_199 + attn_302
        add_199 = attn_302 = None
        attn_304 = torch.conv2d(
            attn_303,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_303 = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_509 = attn_304 * u_33
        attn_304 = u_33 = None
        x_510 = torch.conv2d(
            x_509,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_509 = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_511 = x_510 + shorcut_33
        x_510 = shorcut_33 = None
        mul_100 = unsqueeze_133 * x_511
        unsqueeze_133 = x_511 = None
        x_512 = x_506 + mul_100
        x_506 = mul_100 = None
        unsqueeze_134 = l_self_modules_backbone_modules_block3_modules_25_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_25_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_135 = unsqueeze_134.unsqueeze(-1)
        unsqueeze_134 = None
        batch_norm_71 = torch.nn.functional.batch_norm(
            x_512,
            l_self_modules_backbone_modules_block3_modules_25_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_25_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_25_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_25_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_25_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_25_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_norm2_parameters_bias_ = (None)
        x_513 = torch.conv2d(
            batch_norm_71,
            l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_71 = l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_514 = torch.conv2d(
            x_513,
            l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_513 = l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_515 = torch._C._nn.gelu(x_514, approximate="none")
        x_514 = None
        x_516 = torch.nn.functional.dropout(x_515, 0.0, False, False)
        x_515 = None
        x_517 = torch.conv2d(
            x_516,
            l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_516 = l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_25_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_518 = torch.nn.functional.dropout(x_517, 0.0, False, False)
        x_517 = None
        mul_101 = unsqueeze_135 * x_518
        unsqueeze_135 = x_518 = None
        x_519 = x_512 + mul_101
        x_512 = mul_101 = None
        view_67 = x_519.view(1, 320, 1024)
        x_519 = None
        x_520 = view_67.permute(0, 2, 1)
        view_67 = None
        permute_70 = x_520.permute(0, 2, 1)
        x_520 = None
        x_521 = permute_70.view(1, 320, 32, 32)
        permute_70 = None
        unsqueeze_136 = l_self_modules_backbone_modules_block3_modules_26_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_26_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_137 = unsqueeze_136.unsqueeze(-1)
        unsqueeze_136 = None
        batch_norm_72 = torch.nn.functional.batch_norm(
            x_521,
            l_self_modules_backbone_modules_block3_modules_26_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_26_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_26_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_26_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_26_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_26_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_norm1_parameters_bias_ = (None)
        shorcut_34 = batch_norm_72.clone()
        x_522 = torch.conv2d(
            batch_norm_72,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_72 = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_523 = torch._C._nn.gelu(x_522, approximate="none")
        x_522 = None
        u_34 = x_523.clone()
        attn_305 = torch.conv2d(
            x_523,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_523 = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_306 = torch.conv2d(
            attn_305,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_307 = torch.conv2d(
            attn_306,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_306 = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_308 = torch.conv2d(
            attn_305,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_309 = torch.conv2d(
            attn_308,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_308 = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_310 = torch.conv2d(
            attn_305,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_311 = torch.conv2d(
            attn_310,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_310 = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_204 = attn_305 + attn_307
        attn_305 = attn_307 = None
        add_205 = add_204 + attn_309
        add_204 = attn_309 = None
        attn_312 = add_205 + attn_311
        add_205 = attn_311 = None
        attn_313 = torch.conv2d(
            attn_312,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_312 = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_524 = attn_313 * u_34
        attn_313 = u_34 = None
        x_525 = torch.conv2d(
            x_524,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_524 = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_526 = x_525 + shorcut_34
        x_525 = shorcut_34 = None
        mul_103 = unsqueeze_137 * x_526
        unsqueeze_137 = x_526 = None
        x_527 = x_521 + mul_103
        x_521 = mul_103 = None
        unsqueeze_138 = l_self_modules_backbone_modules_block3_modules_26_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_26_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_139 = unsqueeze_138.unsqueeze(-1)
        unsqueeze_138 = None
        batch_norm_73 = torch.nn.functional.batch_norm(
            x_527,
            l_self_modules_backbone_modules_block3_modules_26_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_26_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_26_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_26_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_26_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_26_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_norm2_parameters_bias_ = (None)
        x_528 = torch.conv2d(
            batch_norm_73,
            l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_73 = l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_529 = torch.conv2d(
            x_528,
            l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_528 = l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_530 = torch._C._nn.gelu(x_529, approximate="none")
        x_529 = None
        x_531 = torch.nn.functional.dropout(x_530, 0.0, False, False)
        x_530 = None
        x_532 = torch.conv2d(
            x_531,
            l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_531 = l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_26_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_533 = torch.nn.functional.dropout(x_532, 0.0, False, False)
        x_532 = None
        mul_104 = unsqueeze_139 * x_533
        unsqueeze_139 = x_533 = None
        x_534 = x_527 + mul_104
        x_527 = mul_104 = None
        view_69 = x_534.view(1, 320, 1024)
        x_534 = None
        x_535 = view_69.permute(0, 2, 1)
        view_69 = None
        x_536 = torch.nn.functional.layer_norm(
            x_535,
            (320,),
            l_self_modules_backbone_modules_norm3_parameters_weight_,
            l_self_modules_backbone_modules_norm3_parameters_bias_,
            1e-05,
        )
        x_535 = (
            l_self_modules_backbone_modules_norm3_parameters_weight_
        ) = l_self_modules_backbone_modules_norm3_parameters_bias_ = None
        reshape_2 = x_536.reshape(1, 32, 32, -1)
        x_536 = None
        permute_72 = reshape_2.permute(0, 3, 1, 2)
        reshape_2 = None
        x_537 = permute_72.contiguous()
        permute_72 = None
        x_538 = torch.conv2d(
            x_537,
            l_self_modules_backbone_modules_patch_embed4_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed4_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_patch_embed4_modules_proj_parameters_weight_ = (
            l_self_modules_backbone_modules_patch_embed4_modules_proj_parameters_bias_
        ) = None
        x_539 = torch.nn.functional.batch_norm(
            x_538,
            l_self_modules_backbone_modules_patch_embed4_modules_norm_buffers_running_mean_,
            l_self_modules_backbone_modules_patch_embed4_modules_norm_buffers_running_var_,
            l_self_modules_backbone_modules_patch_embed4_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed4_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_538 = l_self_modules_backbone_modules_patch_embed4_modules_norm_buffers_running_mean_ = l_self_modules_backbone_modules_patch_embed4_modules_norm_buffers_running_var_ = (
            l_self_modules_backbone_modules_patch_embed4_modules_norm_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_patch_embed4_modules_norm_parameters_bias_
        ) = None
        flatten_3 = x_539.flatten(2)
        x_539 = None
        x_540 = flatten_3.transpose(1, 2)
        flatten_3 = None
        permute_73 = x_540.permute(0, 2, 1)
        x_540 = None
        x_541 = permute_73.view(1, 512, 16, 16)
        permute_73 = None
        unsqueeze_140 = l_self_modules_backbone_modules_block4_modules_0_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block4_modules_0_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_141 = unsqueeze_140.unsqueeze(-1)
        unsqueeze_140 = None
        batch_norm_75 = torch.nn.functional.batch_norm(
            x_541,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block4_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block4_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block4_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_norm1_parameters_bias_ = (None)
        shorcut_35 = batch_norm_75.clone()
        x_542 = torch.conv2d(
            batch_norm_75,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_75 = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_543 = torch._C._nn.gelu(x_542, approximate="none")
        x_542 = None
        u_35 = x_543.clone()
        attn_314 = torch.conv2d(
            x_543,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        x_543 = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_315 = torch.conv2d(
            attn_314,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_316 = torch.conv2d(
            attn_315,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            512,
        )
        attn_315 = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_317 = torch.conv2d(
            attn_314,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_318 = torch.conv2d(
            attn_317,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            512,
        )
        attn_317 = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_319 = torch.conv2d(
            attn_314,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_320 = torch.conv2d(
            attn_319,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            512,
        )
        attn_319 = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_210 = attn_314 + attn_316
        attn_314 = attn_316 = None
        add_211 = add_210 + attn_318
        add_210 = attn_318 = None
        attn_321 = add_211 + attn_320
        add_211 = attn_320 = None
        attn_322 = torch.conv2d(
            attn_321,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_321 = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_544 = attn_322 * u_35
        attn_322 = u_35 = None
        x_545 = torch.conv2d(
            x_544,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_544 = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_546 = x_545 + shorcut_35
        x_545 = shorcut_35 = None
        mul_106 = unsqueeze_141 * x_546
        unsqueeze_141 = x_546 = None
        x_547 = x_541 + mul_106
        x_541 = mul_106 = None
        unsqueeze_142 = l_self_modules_backbone_modules_block4_modules_0_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block4_modules_0_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_143 = unsqueeze_142.unsqueeze(-1)
        unsqueeze_142 = None
        batch_norm_76 = torch.nn.functional.batch_norm(
            x_547,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block4_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block4_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block4_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_norm2_parameters_bias_ = (None)
        x_548 = torch.conv2d(
            batch_norm_76,
            l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_76 = l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_549 = torch.conv2d(
            x_548,
            l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_548 = l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_550 = torch._C._nn.gelu(x_549, approximate="none")
        x_549 = None
        x_551 = torch.nn.functional.dropout(x_550, 0.0, False, False)
        x_550 = None
        x_552 = torch.conv2d(
            x_551,
            l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_551 = l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_553 = torch.nn.functional.dropout(x_552, 0.0, False, False)
        x_552 = None
        mul_107 = unsqueeze_143 * x_553
        unsqueeze_143 = x_553 = None
        x_554 = x_547 + mul_107
        x_547 = mul_107 = None
        view_71 = x_554.view(1, 512, 256)
        x_554 = None
        x_555 = view_71.permute(0, 2, 1)
        view_71 = None
        permute_75 = x_555.permute(0, 2, 1)
        x_555 = None
        x_556 = permute_75.view(1, 512, 16, 16)
        permute_75 = None
        unsqueeze_144 = l_self_modules_backbone_modules_block4_modules_1_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block4_modules_1_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_145 = unsqueeze_144.unsqueeze(-1)
        unsqueeze_144 = None
        batch_norm_77 = torch.nn.functional.batch_norm(
            x_556,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block4_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block4_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block4_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_norm1_parameters_bias_ = (None)
        shorcut_36 = batch_norm_77.clone()
        x_557 = torch.conv2d(
            batch_norm_77,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_77 = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_558 = torch._C._nn.gelu(x_557, approximate="none")
        x_557 = None
        u_36 = x_558.clone()
        attn_323 = torch.conv2d(
            x_558,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        x_558 = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_324 = torch.conv2d(
            attn_323,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_325 = torch.conv2d(
            attn_324,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            512,
        )
        attn_324 = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_326 = torch.conv2d(
            attn_323,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_327 = torch.conv2d(
            attn_326,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            512,
        )
        attn_326 = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_328 = torch.conv2d(
            attn_323,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_329 = torch.conv2d(
            attn_328,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            512,
        )
        attn_328 = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_216 = attn_323 + attn_325
        attn_323 = attn_325 = None
        add_217 = add_216 + attn_327
        add_216 = attn_327 = None
        attn_330 = add_217 + attn_329
        add_217 = attn_329 = None
        attn_331 = torch.conv2d(
            attn_330,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_330 = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_559 = attn_331 * u_36
        attn_331 = u_36 = None
        x_560 = torch.conv2d(
            x_559,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_559 = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_561 = x_560 + shorcut_36
        x_560 = shorcut_36 = None
        mul_109 = unsqueeze_145 * x_561
        unsqueeze_145 = x_561 = None
        x_562 = x_556 + mul_109
        x_556 = mul_109 = None
        unsqueeze_146 = l_self_modules_backbone_modules_block4_modules_1_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block4_modules_1_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_147 = unsqueeze_146.unsqueeze(-1)
        unsqueeze_146 = None
        batch_norm_78 = torch.nn.functional.batch_norm(
            x_562,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block4_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block4_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block4_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_norm2_parameters_bias_ = (None)
        x_563 = torch.conv2d(
            batch_norm_78,
            l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_78 = l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_564 = torch.conv2d(
            x_563,
            l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_563 = l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_565 = torch._C._nn.gelu(x_564, approximate="none")
        x_564 = None
        x_566 = torch.nn.functional.dropout(x_565, 0.0, False, False)
        x_565 = None
        x_567 = torch.conv2d(
            x_566,
            l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_566 = l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_568 = torch.nn.functional.dropout(x_567, 0.0, False, False)
        x_567 = None
        mul_110 = unsqueeze_147 * x_568
        unsqueeze_147 = x_568 = None
        x_569 = x_562 + mul_110
        x_562 = mul_110 = None
        view_73 = x_569.view(1, 512, 256)
        x_569 = None
        x_570 = view_73.permute(0, 2, 1)
        view_73 = None
        permute_77 = x_570.permute(0, 2, 1)
        x_570 = None
        x_571 = permute_77.view(1, 512, 16, 16)
        permute_77 = None
        unsqueeze_148 = l_self_modules_backbone_modules_block4_modules_2_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block4_modules_2_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_149 = unsqueeze_148.unsqueeze(-1)
        unsqueeze_148 = None
        batch_norm_79 = torch.nn.functional.batch_norm(
            x_571,
            l_self_modules_backbone_modules_block4_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block4_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block4_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block4_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block4_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block4_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_norm1_parameters_bias_ = (None)
        shorcut_37 = batch_norm_79.clone()
        x_572 = torch.conv2d(
            batch_norm_79,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_79 = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_573 = torch._C._nn.gelu(x_572, approximate="none")
        x_572 = None
        u_37 = x_573.clone()
        attn_332 = torch.conv2d(
            x_573,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        x_573 = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_333 = torch.conv2d(
            attn_332,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_334 = torch.conv2d(
            attn_333,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            512,
        )
        attn_333 = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_335 = torch.conv2d(
            attn_332,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_336 = torch.conv2d(
            attn_335,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            512,
        )
        attn_335 = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_337 = torch.conv2d(
            attn_332,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_338 = torch.conv2d(
            attn_337,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            512,
        )
        attn_337 = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_222 = attn_332 + attn_334
        attn_332 = attn_334 = None
        add_223 = add_222 + attn_336
        add_222 = attn_336 = None
        attn_339 = add_223 + attn_338
        add_223 = attn_338 = None
        attn_340 = torch.conv2d(
            attn_339,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_339 = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_574 = attn_340 * u_37
        attn_340 = u_37 = None
        x_575 = torch.conv2d(
            x_574,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_574 = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_576 = x_575 + shorcut_37
        x_575 = shorcut_37 = None
        mul_112 = unsqueeze_149 * x_576
        unsqueeze_149 = x_576 = None
        x_577 = x_571 + mul_112
        x_571 = mul_112 = None
        unsqueeze_150 = l_self_modules_backbone_modules_block4_modules_2_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block4_modules_2_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_151 = unsqueeze_150.unsqueeze(-1)
        unsqueeze_150 = None
        batch_norm_80 = torch.nn.functional.batch_norm(
            x_577,
            l_self_modules_backbone_modules_block4_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block4_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block4_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block4_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block4_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block4_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_norm2_parameters_bias_ = (None)
        x_578 = torch.conv2d(
            batch_norm_80,
            l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_80 = l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_579 = torch.conv2d(
            x_578,
            l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_578 = l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_580 = torch._C._nn.gelu(x_579, approximate="none")
        x_579 = None
        x_581 = torch.nn.functional.dropout(x_580, 0.0, False, False)
        x_580 = None
        x_582 = torch.conv2d(
            x_581,
            l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_581 = l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_583 = torch.nn.functional.dropout(x_582, 0.0, False, False)
        x_582 = None
        mul_113 = unsqueeze_151 * x_583
        unsqueeze_151 = x_583 = None
        x_584 = x_577 + mul_113
        x_577 = mul_113 = None
        view_75 = x_584.view(1, 512, 256)
        x_584 = None
        x_585 = view_75.permute(0, 2, 1)
        view_75 = None
        x_586 = torch.nn.functional.layer_norm(
            x_585,
            (512,),
            l_self_modules_backbone_modules_norm4_parameters_weight_,
            l_self_modules_backbone_modules_norm4_parameters_bias_,
            1e-05,
        )
        x_585 = (
            l_self_modules_backbone_modules_norm4_parameters_weight_
        ) = l_self_modules_backbone_modules_norm4_parameters_bias_ = None
        reshape_3 = x_586.reshape(1, 16, 16, -1)
        x_586 = None
        permute_79 = reshape_3.permute(0, 3, 1, 2)
        reshape_3 = None
        x_587 = permute_79.contiguous()
        permute_79 = None
        interpolate = torch.nn.functional.interpolate(
            x_127, (64, 64), None, "bilinear", False
        )
        x_127 = None
        interpolate_1 = torch.nn.functional.interpolate(
            x_537, (64, 64), None, "bilinear", False
        )
        x_537 = None
        interpolate_2 = torch.nn.functional.interpolate(
            x_587, (64, 64), None, "bilinear", False
        )
        x_587 = None
        cat = torch.cat([interpolate, interpolate_1, interpolate_2], dim=1)
        interpolate = interpolate_1 = interpolate_2 = None
        x_588 = torch.conv2d(
            cat,
            l_self_modules_decode_head_modules_squeeze_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat = (
            l_self_modules_decode_head_modules_squeeze_modules_conv_parameters_weight_
        ) = None
        x_589 = torch.nn.functional.group_norm(
            x_588,
            32,
            l_self_modules_decode_head_modules_squeeze_modules_gn_parameters_weight_,
            l_self_modules_decode_head_modules_squeeze_modules_gn_parameters_bias_,
            1e-05,
        )
        x_588 = (
            l_self_modules_decode_head_modules_squeeze_modules_gn_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_squeeze_modules_gn_parameters_bias_
        ) = None
        x_590 = torch.nn.functional.relu(x_589, inplace=True)
        x_589 = None
        x_591 = torch.conv2d(
            x_590,
            l_self_modules_decode_head_modules_hamburger_modules_ham_in_modules_conv_parameters_weight_,
            l_self_modules_decode_head_modules_hamburger_modules_ham_in_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_decode_head_modules_hamburger_modules_ham_in_modules_conv_parameters_weight_ = l_self_modules_decode_head_modules_hamburger_modules_ham_in_modules_conv_parameters_bias_ = (None)
        enjoy = torch.nn.functional.relu(x_591, inplace=True)
        x_591 = None
        x_592 = enjoy.view(1, 1024, 4096)
        enjoy = None
        rand = torch.rand((1, 1024, 16))
        bases = rand.to(device(type="cuda", index=0))
        rand = None
        bases_1 = torch.nn.functional.normalize(bases, dim=1)
        bases = None
        transpose_4 = x_592.transpose(1, 2)
        coef = torch.bmm(transpose_4, bases_1)
        transpose_4 = None
        mul_114 = 1 * coef
        coef = None
        coef_1 = torch.nn.functional.softmax(mul_114, dim=-1)
        mul_114 = None
        transpose_5 = x_592.transpose(1, 2)
        numerator = torch.bmm(transpose_5, bases_1)
        transpose_5 = None
        transpose_6 = bases_1.transpose(1, 2)
        bmm_2 = transpose_6.bmm(bases_1)
        transpose_6 = None
        denominator = coef_1.bmm(bmm_2)
        bmm_2 = None
        mul_115 = coef_1 * numerator
        coef_1 = numerator = None
        add_228 = denominator + 1e-06
        denominator = None
        coef_2 = mul_115 / add_228
        mul_115 = add_228 = None
        numerator_1 = torch.bmm(x_592, coef_2)
        transpose_7 = coef_2.transpose(1, 2)
        bmm_5 = transpose_7.bmm(coef_2)
        transpose_7 = None
        denominator_1 = bases_1.bmm(bmm_5)
        bmm_5 = None
        mul_116 = bases_1 * numerator_1
        bases_1 = numerator_1 = None
        add_229 = denominator_1 + 1e-06
        denominator_1 = None
        bases_2 = mul_116 / add_229
        mul_116 = add_229 = None
        transpose_8 = x_592.transpose(1, 2)
        numerator_2 = torch.bmm(transpose_8, bases_2)
        transpose_8 = None
        transpose_9 = bases_2.transpose(1, 2)
        bmm_8 = transpose_9.bmm(bases_2)
        transpose_9 = None
        denominator_2 = coef_2.bmm(bmm_8)
        bmm_8 = None
        mul_117 = coef_2 * numerator_2
        coef_2 = numerator_2 = None
        add_230 = denominator_2 + 1e-06
        denominator_2 = None
        coef_3 = mul_117 / add_230
        mul_117 = add_230 = None
        numerator_3 = torch.bmm(x_592, coef_3)
        transpose_10 = coef_3.transpose(1, 2)
        bmm_11 = transpose_10.bmm(coef_3)
        transpose_10 = None
        denominator_3 = bases_2.bmm(bmm_11)
        bmm_11 = None
        mul_118 = bases_2 * numerator_3
        bases_2 = numerator_3 = None
        add_231 = denominator_3 + 1e-06
        denominator_3 = None
        bases_3 = mul_118 / add_231
        mul_118 = add_231 = None
        transpose_11 = x_592.transpose(1, 2)
        numerator_4 = torch.bmm(transpose_11, bases_3)
        transpose_11 = None
        transpose_12 = bases_3.transpose(1, 2)
        bmm_14 = transpose_12.bmm(bases_3)
        transpose_12 = None
        denominator_4 = coef_3.bmm(bmm_14)
        bmm_14 = None
        mul_119 = coef_3 * numerator_4
        coef_3 = numerator_4 = None
        add_232 = denominator_4 + 1e-06
        denominator_4 = None
        coef_4 = mul_119 / add_232
        mul_119 = add_232 = None
        numerator_5 = torch.bmm(x_592, coef_4)
        transpose_13 = coef_4.transpose(1, 2)
        bmm_17 = transpose_13.bmm(coef_4)
        transpose_13 = None
        denominator_5 = bases_3.bmm(bmm_17)
        bmm_17 = None
        mul_120 = bases_3 * numerator_5
        bases_3 = numerator_5 = None
        add_233 = denominator_5 + 1e-06
        denominator_5 = None
        bases_4 = mul_120 / add_233
        mul_120 = add_233 = None
        transpose_14 = x_592.transpose(1, 2)
        numerator_6 = torch.bmm(transpose_14, bases_4)
        transpose_14 = None
        transpose_15 = bases_4.transpose(1, 2)
        bmm_20 = transpose_15.bmm(bases_4)
        transpose_15 = None
        denominator_6 = coef_4.bmm(bmm_20)
        bmm_20 = None
        mul_121 = coef_4 * numerator_6
        coef_4 = numerator_6 = None
        add_234 = denominator_6 + 1e-06
        denominator_6 = None
        coef_5 = mul_121 / add_234
        mul_121 = add_234 = None
        numerator_7 = torch.bmm(x_592, coef_5)
        transpose_16 = coef_5.transpose(1, 2)
        bmm_23 = transpose_16.bmm(coef_5)
        transpose_16 = None
        denominator_7 = bases_4.bmm(bmm_23)
        bmm_23 = None
        mul_122 = bases_4 * numerator_7
        bases_4 = numerator_7 = None
        add_235 = denominator_7 + 1e-06
        denominator_7 = None
        bases_5 = mul_122 / add_235
        mul_122 = add_235 = None
        transpose_17 = x_592.transpose(1, 2)
        numerator_8 = torch.bmm(transpose_17, bases_5)
        transpose_17 = None
        transpose_18 = bases_5.transpose(1, 2)
        bmm_26 = transpose_18.bmm(bases_5)
        transpose_18 = None
        denominator_8 = coef_5.bmm(bmm_26)
        bmm_26 = None
        mul_123 = coef_5 * numerator_8
        coef_5 = numerator_8 = None
        add_236 = denominator_8 + 1e-06
        denominator_8 = None
        coef_6 = mul_123 / add_236
        mul_123 = add_236 = None
        numerator_9 = torch.bmm(x_592, coef_6)
        transpose_19 = coef_6.transpose(1, 2)
        bmm_29 = transpose_19.bmm(coef_6)
        transpose_19 = None
        denominator_9 = bases_5.bmm(bmm_29)
        bmm_29 = None
        mul_124 = bases_5 * numerator_9
        bases_5 = numerator_9 = None
        add_237 = denominator_9 + 1e-06
        denominator_9 = None
        bases_6 = mul_124 / add_237
        mul_124 = add_237 = None
        transpose_20 = x_592.transpose(1, 2)
        numerator_10 = torch.bmm(transpose_20, bases_6)
        transpose_20 = None
        transpose_21 = bases_6.transpose(1, 2)
        bmm_32 = transpose_21.bmm(bases_6)
        transpose_21 = None
        denominator_10 = coef_6.bmm(bmm_32)
        bmm_32 = None
        mul_125 = coef_6 * numerator_10
        coef_6 = numerator_10 = None
        add_238 = denominator_10 + 1e-06
        denominator_10 = None
        coef_7 = mul_125 / add_238
        mul_125 = add_238 = None
        numerator_11 = torch.bmm(x_592, coef_7)
        transpose_22 = coef_7.transpose(1, 2)
        bmm_35 = transpose_22.bmm(coef_7)
        transpose_22 = None
        denominator_11 = bases_6.bmm(bmm_35)
        bmm_35 = None
        mul_126 = bases_6 * numerator_11
        bases_6 = numerator_11 = None
        add_239 = denominator_11 + 1e-06
        denominator_11 = None
        bases_7 = mul_126 / add_239
        mul_126 = add_239 = None
        transpose_23 = x_592.transpose(1, 2)
        numerator_12 = torch.bmm(transpose_23, bases_7)
        transpose_23 = None
        transpose_24 = bases_7.transpose(1, 2)
        bmm_38 = transpose_24.bmm(bases_7)
        transpose_24 = None
        denominator_12 = coef_7.bmm(bmm_38)
        bmm_38 = None
        mul_127 = coef_7 * numerator_12
        coef_7 = numerator_12 = None
        add_240 = denominator_12 + 1e-06
        denominator_12 = None
        coef_8 = mul_127 / add_240
        mul_127 = add_240 = None
        numerator_13 = torch.bmm(x_592, coef_8)
        transpose_25 = coef_8.transpose(1, 2)
        bmm_41 = transpose_25.bmm(coef_8)
        transpose_25 = None
        denominator_13 = bases_7.bmm(bmm_41)
        bmm_41 = None
        mul_128 = bases_7 * numerator_13
        bases_7 = numerator_13 = None
        add_241 = denominator_13 + 1e-06
        denominator_13 = None
        bases_8 = mul_128 / add_241
        mul_128 = add_241 = None
        transpose_26 = x_592.transpose(1, 2)
        x_592 = None
        numerator_14 = torch.bmm(transpose_26, bases_8)
        transpose_26 = None
        transpose_27 = bases_8.transpose(1, 2)
        bmm_44 = transpose_27.bmm(bases_8)
        transpose_27 = None
        denominator_14 = coef_8.bmm(bmm_44)
        bmm_44 = None
        mul_129 = coef_8 * numerator_14
        coef_8 = numerator_14 = None
        add_242 = denominator_14 + 1e-06
        denominator_14 = None
        coef_9 = mul_129 / add_242
        mul_129 = add_242 = None
        transpose_28 = coef_9.transpose(1, 2)
        coef_9 = None
        x_593 = torch.bmm(bases_8, transpose_28)
        bases_8 = transpose_28 = None
        x_594 = x_593.view(1, 1024, 64, 64)
        x_593 = None
        x_595 = torch.conv2d(
            x_594,
            l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_594 = l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_conv_parameters_weight_ = (None)
        x_596 = torch.nn.functional.group_norm(
            x_595,
            32,
            l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_gn_parameters_weight_,
            l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_gn_parameters_bias_,
            1e-05,
        )
        x_595 = l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_gn_parameters_weight_ = l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_gn_parameters_bias_ = (None)
        add_243 = x_590 + x_596
        x_590 = x_596 = None
        ham = torch.nn.functional.relu(add_243, inplace=True)
        add_243 = None
        x_597 = torch.conv2d(
            ham,
            l_self_modules_decode_head_modules_align_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ham = (
            l_self_modules_decode_head_modules_align_modules_conv_parameters_weight_
        ) = None
        x_598 = torch.nn.functional.group_norm(
            x_597,
            32,
            l_self_modules_decode_head_modules_align_modules_gn_parameters_weight_,
            l_self_modules_decode_head_modules_align_modules_gn_parameters_bias_,
            1e-05,
        )
        x_597 = (
            l_self_modules_decode_head_modules_align_modules_gn_parameters_weight_
        ) = l_self_modules_decode_head_modules_align_modules_gn_parameters_bias_ = None
        x_599 = torch.nn.functional.relu(x_598, inplace=True)
        x_598 = None
        feat = torch.nn.functional.dropout2d(x_599, 0.1, False, False)
        x_599 = None
        output = torch.conv2d(
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
        return (output,)
