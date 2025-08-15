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
        x_31 = torch.nn.functional.layer_norm(
            x_30,
            (64,),
            l_self_modules_backbone_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_30 = (
            l_self_modules_backbone_modules_norm1_parameters_weight_
        ) = l_self_modules_backbone_modules_norm1_parameters_bias_ = None
        reshape = x_31.reshape(1, 128, 128, -1)
        x_31 = None
        permute_4 = reshape.permute(0, 3, 1, 2)
        reshape = None
        x_32 = permute_4.contiguous()
        permute_4 = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_backbone_modules_patch_embed2_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed2_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_32 = (
            l_self_modules_backbone_modules_patch_embed2_modules_proj_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_patch_embed2_modules_proj_parameters_bias_
        ) = None
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_backbone_modules_patch_embed2_modules_norm_buffers_running_mean_,
            l_self_modules_backbone_modules_patch_embed2_modules_norm_buffers_running_var_,
            l_self_modules_backbone_modules_patch_embed2_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_backbone_modules_patch_embed2_modules_norm_buffers_running_mean_ = l_self_modules_backbone_modules_patch_embed2_modules_norm_buffers_running_var_ = (
            l_self_modules_backbone_modules_patch_embed2_modules_norm_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_patch_embed2_modules_norm_parameters_bias_
        ) = None
        flatten_1 = x_34.flatten(2)
        x_34 = None
        x_35 = flatten_1.transpose(1, 2)
        flatten_1 = None
        permute_5 = x_35.permute(0, 2, 1)
        x_35 = None
        x_36 = permute_5.view(1, 128, 64, 64)
        permute_5 = None
        unsqueeze_8 = l_self_modules_backbone_modules_block2_modules_0_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block2_modules_0_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_9 = unsqueeze_8.unsqueeze(-1)
        unsqueeze_8 = None
        batch_norm_7 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block2_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block2_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block2_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_norm1_parameters_bias_ = (None)
        shorcut_2 = batch_norm_7.clone()
        x_37 = torch.conv2d(
            batch_norm_7,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_7 = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_38 = torch._C._nn.gelu(x_37, approximate="none")
        x_37 = None
        u_2 = x_38.clone()
        attn_17 = torch.conv2d(
            x_38,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            128,
        )
        x_38 = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_18 = torch.conv2d(
            attn_17,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_19 = torch.conv2d(
            attn_18,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            128,
        )
        attn_18 = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_20 = torch.conv2d(
            attn_17,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_21 = torch.conv2d(
            attn_20,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            128,
        )
        attn_20 = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_22 = torch.conv2d(
            attn_17,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_23 = torch.conv2d(
            attn_22,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            128,
        )
        attn_22 = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_12 = attn_17 + attn_19
        attn_17 = attn_19 = None
        add_13 = add_12 + attn_21
        add_12 = attn_21 = None
        attn_24 = add_13 + attn_23
        add_13 = attn_23 = None
        attn_25 = torch.conv2d(
            attn_24,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_24 = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_39 = attn_25 * u_2
        attn_25 = u_2 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_41 = x_40 + shorcut_2
        x_40 = shorcut_2 = None
        mul_7 = unsqueeze_9 * x_41
        unsqueeze_9 = x_41 = None
        x_42 = x_36 + mul_7
        x_36 = mul_7 = None
        unsqueeze_10 = l_self_modules_backbone_modules_block2_modules_0_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block2_modules_0_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_11 = unsqueeze_10.unsqueeze(-1)
        unsqueeze_10 = None
        batch_norm_8 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block2_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block2_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block2_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_norm2_parameters_bias_ = (None)
        x_43 = torch.conv2d(
            batch_norm_8,
            l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_8 = l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_43 = l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_45 = torch._C._nn.gelu(x_44, approximate="none")
        x_44 = None
        x_46 = torch.nn.functional.dropout(x_45, 0.0, False, False)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_48 = torch.nn.functional.dropout(x_47, 0.0, False, False)
        x_47 = None
        mul_8 = unsqueeze_11 * x_48
        unsqueeze_11 = x_48 = None
        x_49 = x_42 + mul_8
        x_42 = mul_8 = None
        view_5 = x_49.view(1, 128, 4096)
        x_49 = None
        x_50 = view_5.permute(0, 2, 1)
        view_5 = None
        permute_7 = x_50.permute(0, 2, 1)
        x_50 = None
        x_51 = permute_7.view(1, 128, 64, 64)
        permute_7 = None
        unsqueeze_12 = l_self_modules_backbone_modules_block2_modules_1_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block2_modules_1_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_13 = unsqueeze_12.unsqueeze(-1)
        unsqueeze_12 = None
        batch_norm_9 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block2_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block2_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block2_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_norm1_parameters_bias_ = (None)
        shorcut_3 = batch_norm_9.clone()
        x_52 = torch.conv2d(
            batch_norm_9,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_9 = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_53 = torch._C._nn.gelu(x_52, approximate="none")
        x_52 = None
        u_3 = x_53.clone()
        attn_26 = torch.conv2d(
            x_53,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            128,
        )
        x_53 = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_27 = torch.conv2d(
            attn_26,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_28 = torch.conv2d(
            attn_27,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            128,
        )
        attn_27 = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_29 = torch.conv2d(
            attn_26,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_30 = torch.conv2d(
            attn_29,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            128,
        )
        attn_29 = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_31 = torch.conv2d(
            attn_26,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_32 = torch.conv2d(
            attn_31,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            128,
        )
        attn_31 = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_18 = attn_26 + attn_28
        attn_26 = attn_28 = None
        add_19 = add_18 + attn_30
        add_18 = attn_30 = None
        attn_33 = add_19 + attn_32
        add_19 = attn_32 = None
        attn_34 = torch.conv2d(
            attn_33,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_33 = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_54 = attn_34 * u_3
        attn_34 = u_3 = None
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_54 = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_56 = x_55 + shorcut_3
        x_55 = shorcut_3 = None
        mul_10 = unsqueeze_13 * x_56
        unsqueeze_13 = x_56 = None
        x_57 = x_51 + mul_10
        x_51 = mul_10 = None
        unsqueeze_14 = l_self_modules_backbone_modules_block2_modules_1_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block2_modules_1_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_15 = unsqueeze_14.unsqueeze(-1)
        unsqueeze_14 = None
        batch_norm_10 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block2_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block2_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block2_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_norm2_parameters_bias_ = (None)
        x_58 = torch.conv2d(
            batch_norm_10,
            l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_10 = l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_58 = l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_60 = torch._C._nn.gelu(x_59, approximate="none")
        x_59 = None
        x_61 = torch.nn.functional.dropout(x_60, 0.0, False, False)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block2_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
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
        x_66 = torch.nn.functional.layer_norm(
            x_65,
            (128,),
            l_self_modules_backbone_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_65 = (
            l_self_modules_backbone_modules_norm2_parameters_weight_
        ) = l_self_modules_backbone_modules_norm2_parameters_bias_ = None
        reshape_1 = x_66.reshape(1, 64, 64, -1)
        x_66 = None
        permute_9 = reshape_1.permute(0, 3, 1, 2)
        reshape_1 = None
        x_67 = permute_9.contiguous()
        permute_9 = None
        x_68 = torch.conv2d(
            x_67,
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
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_backbone_modules_patch_embed3_modules_norm_buffers_running_mean_,
            l_self_modules_backbone_modules_patch_embed3_modules_norm_buffers_running_var_,
            l_self_modules_backbone_modules_patch_embed3_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_backbone_modules_patch_embed3_modules_norm_buffers_running_mean_ = l_self_modules_backbone_modules_patch_embed3_modules_norm_buffers_running_var_ = (
            l_self_modules_backbone_modules_patch_embed3_modules_norm_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_patch_embed3_modules_norm_parameters_bias_
        ) = None
        flatten_2 = x_69.flatten(2)
        x_69 = None
        x_70 = flatten_2.transpose(1, 2)
        flatten_2 = None
        permute_10 = x_70.permute(0, 2, 1)
        x_70 = None
        x_71 = permute_10.view(1, 320, 32, 32)
        permute_10 = None
        unsqueeze_16 = l_self_modules_backbone_modules_block3_modules_0_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_0_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_17 = unsqueeze_16.unsqueeze(-1)
        unsqueeze_16 = None
        batch_norm_12 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_norm1_parameters_bias_ = (None)
        shorcut_4 = batch_norm_12.clone()
        x_72 = torch.conv2d(
            batch_norm_12,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_12 = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_73 = torch._C._nn.gelu(x_72, approximate="none")
        x_72 = None
        u_4 = x_73.clone()
        attn_35 = torch.conv2d(
            x_73,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_73 = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_36 = torch.conv2d(
            attn_35,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_37 = torch.conv2d(
            attn_36,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_36 = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_38 = torch.conv2d(
            attn_35,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_39 = torch.conv2d(
            attn_38,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_38 = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_40 = torch.conv2d(
            attn_35,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_41 = torch.conv2d(
            attn_40,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_40 = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_24 = attn_35 + attn_37
        attn_35 = attn_37 = None
        add_25 = add_24 + attn_39
        add_24 = attn_39 = None
        attn_42 = add_25 + attn_41
        add_25 = attn_41 = None
        attn_43 = torch.conv2d(
            attn_42,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_42 = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_74 = attn_43 * u_4
        attn_43 = u_4 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_76 = x_75 + shorcut_4
        x_75 = shorcut_4 = None
        mul_13 = unsqueeze_17 * x_76
        unsqueeze_17 = x_76 = None
        x_77 = x_71 + mul_13
        x_71 = mul_13 = None
        unsqueeze_18 = l_self_modules_backbone_modules_block3_modules_0_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_0_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_19 = unsqueeze_18.unsqueeze(-1)
        unsqueeze_18 = None
        batch_norm_13 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_norm2_parameters_bias_ = (None)
        x_78 = torch.conv2d(
            batch_norm_13,
            l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_13 = l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_78 = l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_80 = torch._C._nn.gelu(x_79, approximate="none")
        x_79 = None
        x_81 = torch.nn.functional.dropout(x_80, 0.0, False, False)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_83 = torch.nn.functional.dropout(x_82, 0.0, False, False)
        x_82 = None
        mul_14 = unsqueeze_19 * x_83
        unsqueeze_19 = x_83 = None
        x_84 = x_77 + mul_14
        x_77 = mul_14 = None
        view_9 = x_84.view(1, 320, 1024)
        x_84 = None
        x_85 = view_9.permute(0, 2, 1)
        view_9 = None
        permute_12 = x_85.permute(0, 2, 1)
        x_85 = None
        x_86 = permute_12.view(1, 320, 32, 32)
        permute_12 = None
        unsqueeze_20 = l_self_modules_backbone_modules_block3_modules_1_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_1_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_21 = unsqueeze_20.unsqueeze(-1)
        unsqueeze_20 = None
        batch_norm_14 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_norm1_parameters_bias_ = (None)
        shorcut_5 = batch_norm_14.clone()
        x_87 = torch.conv2d(
            batch_norm_14,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_14 = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_88 = torch._C._nn.gelu(x_87, approximate="none")
        x_87 = None
        u_5 = x_88.clone()
        attn_44 = torch.conv2d(
            x_88,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_88 = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_45 = torch.conv2d(
            attn_44,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_46 = torch.conv2d(
            attn_45,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_45 = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_47 = torch.conv2d(
            attn_44,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_48 = torch.conv2d(
            attn_47,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_47 = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_49 = torch.conv2d(
            attn_44,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_50 = torch.conv2d(
            attn_49,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_49 = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_30 = attn_44 + attn_46
        attn_44 = attn_46 = None
        add_31 = add_30 + attn_48
        add_30 = attn_48 = None
        attn_51 = add_31 + attn_50
        add_31 = attn_50 = None
        attn_52 = torch.conv2d(
            attn_51,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_51 = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_89 = attn_52 * u_5
        attn_52 = u_5 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_91 = x_90 + shorcut_5
        x_90 = shorcut_5 = None
        mul_16 = unsqueeze_21 * x_91
        unsqueeze_21 = x_91 = None
        x_92 = x_86 + mul_16
        x_86 = mul_16 = None
        unsqueeze_22 = l_self_modules_backbone_modules_block3_modules_1_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_1_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_23 = unsqueeze_22.unsqueeze(-1)
        unsqueeze_22 = None
        batch_norm_15 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_norm2_parameters_bias_ = (None)
        x_93 = torch.conv2d(
            batch_norm_15,
            l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_15 = l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_93 = l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_95 = torch._C._nn.gelu(x_94, approximate="none")
        x_94 = None
        x_96 = torch.nn.functional.dropout(x_95, 0.0, False, False)
        x_95 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_98 = torch.nn.functional.dropout(x_97, 0.0, False, False)
        x_97 = None
        mul_17 = unsqueeze_23 * x_98
        unsqueeze_23 = x_98 = None
        x_99 = x_92 + mul_17
        x_92 = mul_17 = None
        view_11 = x_99.view(1, 320, 1024)
        x_99 = None
        x_100 = view_11.permute(0, 2, 1)
        view_11 = None
        permute_14 = x_100.permute(0, 2, 1)
        x_100 = None
        x_101 = permute_14.view(1, 320, 32, 32)
        permute_14 = None
        unsqueeze_24 = l_self_modules_backbone_modules_block3_modules_2_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_2_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_25 = unsqueeze_24.unsqueeze(-1)
        unsqueeze_24 = None
        batch_norm_16 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_norm1_parameters_bias_ = (None)
        shorcut_6 = batch_norm_16.clone()
        x_102 = torch.conv2d(
            batch_norm_16,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_16 = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_103 = torch._C._nn.gelu(x_102, approximate="none")
        x_102 = None
        u_6 = x_103.clone()
        attn_53 = torch.conv2d(
            x_103,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_103 = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_54 = torch.conv2d(
            attn_53,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_55 = torch.conv2d(
            attn_54,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_54 = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_56 = torch.conv2d(
            attn_53,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_57 = torch.conv2d(
            attn_56,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_56 = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_58 = torch.conv2d(
            attn_53,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_59 = torch.conv2d(
            attn_58,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_58 = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_36 = attn_53 + attn_55
        attn_53 = attn_55 = None
        add_37 = add_36 + attn_57
        add_36 = attn_57 = None
        attn_60 = add_37 + attn_59
        add_37 = attn_59 = None
        attn_61 = torch.conv2d(
            attn_60,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_60 = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_104 = attn_61 * u_6
        attn_61 = u_6 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_106 = x_105 + shorcut_6
        x_105 = shorcut_6 = None
        mul_19 = unsqueeze_25 * x_106
        unsqueeze_25 = x_106 = None
        x_107 = x_101 + mul_19
        x_101 = mul_19 = None
        unsqueeze_26 = l_self_modules_backbone_modules_block3_modules_2_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_2_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_27 = unsqueeze_26.unsqueeze(-1)
        unsqueeze_26 = None
        batch_norm_17 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_norm2_parameters_bias_ = (None)
        x_108 = torch.conv2d(
            batch_norm_17,
            l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_17 = l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_108 = l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_110 = torch._C._nn.gelu(x_109, approximate="none")
        x_109 = None
        x_111 = torch.nn.functional.dropout(x_110, 0.0, False, False)
        x_110 = None
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_111 = l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_113 = torch.nn.functional.dropout(x_112, 0.0, False, False)
        x_112 = None
        mul_20 = unsqueeze_27 * x_113
        unsqueeze_27 = x_113 = None
        x_114 = x_107 + mul_20
        x_107 = mul_20 = None
        view_13 = x_114.view(1, 320, 1024)
        x_114 = None
        x_115 = view_13.permute(0, 2, 1)
        view_13 = None
        permute_16 = x_115.permute(0, 2, 1)
        x_115 = None
        x_116 = permute_16.view(1, 320, 32, 32)
        permute_16 = None
        unsqueeze_28 = l_self_modules_backbone_modules_block3_modules_3_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_3_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_29 = unsqueeze_28.unsqueeze(-1)
        unsqueeze_28 = None
        batch_norm_18 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_3_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_3_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_3_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_norm1_parameters_bias_ = (None)
        shorcut_7 = batch_norm_18.clone()
        x_117 = torch.conv2d(
            batch_norm_18,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_18 = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_118 = torch._C._nn.gelu(x_117, approximate="none")
        x_117 = None
        u_7 = x_118.clone()
        attn_62 = torch.conv2d(
            x_118,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            320,
        )
        x_118 = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_63 = torch.conv2d(
            attn_62,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_64 = torch.conv2d(
            attn_63,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            320,
        )
        attn_63 = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_65 = torch.conv2d(
            attn_62,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_66 = torch.conv2d(
            attn_65,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            320,
        )
        attn_65 = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_67 = torch.conv2d(
            attn_62,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            320,
        )
        l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_68 = torch.conv2d(
            attn_67,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            320,
        )
        attn_67 = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_42 = attn_62 + attn_64
        attn_62 = attn_64 = None
        add_43 = add_42 + attn_66
        add_42 = attn_66 = None
        attn_69 = add_43 + attn_68
        add_43 = attn_68 = None
        attn_70 = torch.conv2d(
            attn_69,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_69 = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_119 = attn_70 * u_7
        attn_70 = u_7 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_119 = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_121 = x_120 + shorcut_7
        x_120 = shorcut_7 = None
        mul_22 = unsqueeze_29 * x_121
        unsqueeze_29 = x_121 = None
        x_122 = x_116 + mul_22
        x_116 = mul_22 = None
        unsqueeze_30 = l_self_modules_backbone_modules_block3_modules_3_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block3_modules_3_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_31 = unsqueeze_30.unsqueeze(-1)
        unsqueeze_30 = None
        batch_norm_19 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block3_modules_3_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block3_modules_3_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block3_modules_3_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_norm2_parameters_bias_ = (None)
        x_123 = torch.conv2d(
            batch_norm_19,
            l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_19 = l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1280,
        )
        x_123 = l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_125 = torch._C._nn.gelu(x_124, approximate="none")
        x_124 = None
        x_126 = torch.nn.functional.dropout(x_125, 0.0, False, False)
        x_125 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_126 = l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block3_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_128 = torch.nn.functional.dropout(x_127, 0.0, False, False)
        x_127 = None
        mul_23 = unsqueeze_31 * x_128
        unsqueeze_31 = x_128 = None
        x_129 = x_122 + mul_23
        x_122 = mul_23 = None
        view_15 = x_129.view(1, 320, 1024)
        x_129 = None
        x_130 = view_15.permute(0, 2, 1)
        view_15 = None
        x_131 = torch.nn.functional.layer_norm(
            x_130,
            (320,),
            l_self_modules_backbone_modules_norm3_parameters_weight_,
            l_self_modules_backbone_modules_norm3_parameters_bias_,
            1e-05,
        )
        x_130 = (
            l_self_modules_backbone_modules_norm3_parameters_weight_
        ) = l_self_modules_backbone_modules_norm3_parameters_bias_ = None
        reshape_2 = x_131.reshape(1, 32, 32, -1)
        x_131 = None
        permute_18 = reshape_2.permute(0, 3, 1, 2)
        reshape_2 = None
        x_132 = permute_18.contiguous()
        permute_18 = None
        x_133 = torch.conv2d(
            x_132,
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
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_backbone_modules_patch_embed4_modules_norm_buffers_running_mean_,
            l_self_modules_backbone_modules_patch_embed4_modules_norm_buffers_running_var_,
            l_self_modules_backbone_modules_patch_embed4_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed4_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_133 = l_self_modules_backbone_modules_patch_embed4_modules_norm_buffers_running_mean_ = l_self_modules_backbone_modules_patch_embed4_modules_norm_buffers_running_var_ = (
            l_self_modules_backbone_modules_patch_embed4_modules_norm_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_patch_embed4_modules_norm_parameters_bias_
        ) = None
        flatten_3 = x_134.flatten(2)
        x_134 = None
        x_135 = flatten_3.transpose(1, 2)
        flatten_3 = None
        permute_19 = x_135.permute(0, 2, 1)
        x_135 = None
        x_136 = permute_19.view(1, 512, 16, 16)
        permute_19 = None
        unsqueeze_32 = l_self_modules_backbone_modules_block4_modules_0_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block4_modules_0_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_33 = unsqueeze_32.unsqueeze(-1)
        unsqueeze_32 = None
        batch_norm_21 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block4_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block4_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block4_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_norm1_parameters_bias_ = (None)
        shorcut_8 = batch_norm_21.clone()
        x_137 = torch.conv2d(
            batch_norm_21,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_21 = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_138 = torch._C._nn.gelu(x_137, approximate="none")
        x_137 = None
        u_8 = x_138.clone()
        attn_71 = torch.conv2d(
            x_138,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        x_138 = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_72 = torch.conv2d(
            attn_71,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_73 = torch.conv2d(
            attn_72,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            512,
        )
        attn_72 = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_74 = torch.conv2d(
            attn_71,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_75 = torch.conv2d(
            attn_74,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            512,
        )
        attn_74 = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_76 = torch.conv2d(
            attn_71,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_77 = torch.conv2d(
            attn_76,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            512,
        )
        attn_76 = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_48 = attn_71 + attn_73
        attn_71 = attn_73 = None
        add_49 = add_48 + attn_75
        add_48 = attn_75 = None
        attn_78 = add_49 + attn_77
        add_49 = attn_77 = None
        attn_79 = torch.conv2d(
            attn_78,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_78 = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_139 = attn_79 * u_8
        attn_79 = u_8 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_141 = x_140 + shorcut_8
        x_140 = shorcut_8 = None
        mul_25 = unsqueeze_33 * x_141
        unsqueeze_33 = x_141 = None
        x_142 = x_136 + mul_25
        x_136 = mul_25 = None
        unsqueeze_34 = l_self_modules_backbone_modules_block4_modules_0_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block4_modules_0_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_35 = unsqueeze_34.unsqueeze(-1)
        unsqueeze_34 = None
        batch_norm_22 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block4_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block4_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block4_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_norm2_parameters_bias_ = (None)
        x_143 = torch.conv2d(
            batch_norm_22,
            l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_22 = l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_143 = l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_145 = torch._C._nn.gelu(x_144, approximate="none")
        x_144 = None
        x_146 = torch.nn.functional.dropout(x_145, 0.0, False, False)
        x_145 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_146 = l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_148 = torch.nn.functional.dropout(x_147, 0.0, False, False)
        x_147 = None
        mul_26 = unsqueeze_35 * x_148
        unsqueeze_35 = x_148 = None
        x_149 = x_142 + mul_26
        x_142 = mul_26 = None
        view_17 = x_149.view(1, 512, 256)
        x_149 = None
        x_150 = view_17.permute(0, 2, 1)
        view_17 = None
        permute_21 = x_150.permute(0, 2, 1)
        x_150 = None
        x_151 = permute_21.view(1, 512, 16, 16)
        permute_21 = None
        unsqueeze_36 = l_self_modules_backbone_modules_block4_modules_1_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block4_modules_1_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_37 = unsqueeze_36.unsqueeze(-1)
        unsqueeze_36 = None
        batch_norm_23 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block4_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_backbone_modules_block4_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_backbone_modules_block4_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_norm1_parameters_bias_ = (None)
        shorcut_9 = batch_norm_23.clone()
        x_152 = torch.conv2d(
            batch_norm_23,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_23 = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_1_parameters_bias_ = (None)
        x_153 = torch._C._nn.gelu(x_152, approximate="none")
        x_152 = None
        u_9 = x_153.clone()
        attn_80 = torch.conv2d(
            x_153,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        x_153 = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_parameters_bias_ = (None)
        attn_81 = torch.conv2d(
            attn_80,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_,
            (1, 1),
            (0, 3),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_1_parameters_bias_ = (None)
        attn_82 = torch.conv2d(
            attn_81,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_,
            (1, 1),
            (3, 0),
            (1, 1),
            512,
        )
        attn_81 = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv0_2_parameters_bias_ = (None)
        attn_83 = torch.conv2d(
            attn_80,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_1_parameters_bias_ = (None)
        attn_84 = torch.conv2d(
            attn_83,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            512,
        )
        attn_83 = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv1_2_parameters_bias_ = (None)
        attn_85 = torch.conv2d(
            attn_80,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_,
            (1, 1),
            (0, 10),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_1_parameters_bias_ = (None)
        attn_86 = torch.conv2d(
            attn_85,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_,
            (1, 1),
            (10, 0),
            (1, 1),
            512,
        )
        attn_85 = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv2_2_parameters_bias_ = (None)
        add_54 = attn_80 + attn_82
        attn_80 = attn_82 = None
        add_55 = add_54 + attn_84
        add_54 = attn_84 = None
        attn_87 = add_55 + attn_86
        add_55 = attn_86 = None
        attn_88 = torch.conv2d(
            attn_87,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        attn_87 = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_spatial_gating_unit_modules_conv3_parameters_bias_ = (None)
        x_154 = attn_88 * u_9
        attn_88 = u_9 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_attn_modules_proj_2_parameters_bias_ = (None)
        x_156 = x_155 + shorcut_9
        x_155 = shorcut_9 = None
        mul_28 = unsqueeze_37 * x_156
        unsqueeze_37 = x_156 = None
        x_157 = x_151 + mul_28
        x_151 = mul_28 = None
        unsqueeze_38 = l_self_modules_backbone_modules_block4_modules_1_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_block4_modules_1_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_39 = unsqueeze_38.unsqueeze(-1)
        unsqueeze_38 = None
        batch_norm_24 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_block4_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_backbone_modules_block4_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_backbone_modules_block4_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_norm2_parameters_bias_ = (None)
        x_158 = torch.conv2d(
            batch_norm_24,
            l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        batch_norm_24 = l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_159 = torch.conv2d(
            x_158,
            l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_dwconv_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_dwconv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_158 = l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_dwconv_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_dwconv_parameters_bias_ = (None)
        x_160 = torch._C._nn.gelu(x_159, approximate="none")
        x_159 = None
        x_161 = torch.nn.functional.dropout(x_160, 0.0, False, False)
        x_160 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_block4_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_163 = torch.nn.functional.dropout(x_162, 0.0, False, False)
        x_162 = None
        mul_29 = unsqueeze_39 * x_163
        unsqueeze_39 = x_163 = None
        x_164 = x_157 + mul_29
        x_157 = mul_29 = None
        view_19 = x_164.view(1, 512, 256)
        x_164 = None
        x_165 = view_19.permute(0, 2, 1)
        view_19 = None
        x_166 = torch.nn.functional.layer_norm(
            x_165,
            (512,),
            l_self_modules_backbone_modules_norm4_parameters_weight_,
            l_self_modules_backbone_modules_norm4_parameters_bias_,
            1e-05,
        )
        x_165 = (
            l_self_modules_backbone_modules_norm4_parameters_weight_
        ) = l_self_modules_backbone_modules_norm4_parameters_bias_ = None
        reshape_3 = x_166.reshape(1, 16, 16, -1)
        x_166 = None
        permute_23 = reshape_3.permute(0, 3, 1, 2)
        reshape_3 = None
        x_167 = permute_23.contiguous()
        permute_23 = None
        interpolate = torch.nn.functional.interpolate(
            x_67, (64, 64), None, "bilinear", False
        )
        x_67 = None
        interpolate_1 = torch.nn.functional.interpolate(
            x_132, (64, 64), None, "bilinear", False
        )
        x_132 = None
        interpolate_2 = torch.nn.functional.interpolate(
            x_167, (64, 64), None, "bilinear", False
        )
        x_167 = None
        cat = torch.cat([interpolate, interpolate_1, interpolate_2], dim=1)
        interpolate = interpolate_1 = interpolate_2 = None
        x_168 = torch.conv2d(
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
        x_169 = torch.nn.functional.group_norm(
            x_168,
            32,
            l_self_modules_decode_head_modules_squeeze_modules_gn_parameters_weight_,
            l_self_modules_decode_head_modules_squeeze_modules_gn_parameters_bias_,
            1e-05,
        )
        x_168 = (
            l_self_modules_decode_head_modules_squeeze_modules_gn_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_squeeze_modules_gn_parameters_bias_
        ) = None
        x_170 = torch.nn.functional.relu(x_169, inplace=True)
        x_169 = None
        x_171 = torch.conv2d(
            x_170,
            l_self_modules_decode_head_modules_hamburger_modules_ham_in_modules_conv_parameters_weight_,
            l_self_modules_decode_head_modules_hamburger_modules_ham_in_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_decode_head_modules_hamburger_modules_ham_in_modules_conv_parameters_weight_ = l_self_modules_decode_head_modules_hamburger_modules_ham_in_modules_conv_parameters_bias_ = (None)
        enjoy = torch.nn.functional.relu(x_171, inplace=True)
        x_171 = None
        x_172 = enjoy.view(1, 256, 4096)
        enjoy = None
        rand = torch.rand((1, 256, 16))
        bases = rand.to(device(type="cuda", index=0))
        rand = None
        bases_1 = torch.nn.functional.normalize(bases, dim=1)
        bases = None
        transpose_4 = x_172.transpose(1, 2)
        coef = torch.bmm(transpose_4, bases_1)
        transpose_4 = None
        mul_30 = 1 * coef
        coef = None
        coef_1 = torch.nn.functional.softmax(mul_30, dim=-1)
        mul_30 = None
        transpose_5 = x_172.transpose(1, 2)
        numerator = torch.bmm(transpose_5, bases_1)
        transpose_5 = None
        transpose_6 = bases_1.transpose(1, 2)
        bmm_2 = transpose_6.bmm(bases_1)
        transpose_6 = None
        denominator = coef_1.bmm(bmm_2)
        bmm_2 = None
        mul_31 = coef_1 * numerator
        coef_1 = numerator = None
        add_60 = denominator + 1e-06
        denominator = None
        coef_2 = mul_31 / add_60
        mul_31 = add_60 = None
        numerator_1 = torch.bmm(x_172, coef_2)
        transpose_7 = coef_2.transpose(1, 2)
        bmm_5 = transpose_7.bmm(coef_2)
        transpose_7 = None
        denominator_1 = bases_1.bmm(bmm_5)
        bmm_5 = None
        mul_32 = bases_1 * numerator_1
        bases_1 = numerator_1 = None
        add_61 = denominator_1 + 1e-06
        denominator_1 = None
        bases_2 = mul_32 / add_61
        mul_32 = add_61 = None
        transpose_8 = x_172.transpose(1, 2)
        numerator_2 = torch.bmm(transpose_8, bases_2)
        transpose_8 = None
        transpose_9 = bases_2.transpose(1, 2)
        bmm_8 = transpose_9.bmm(bases_2)
        transpose_9 = None
        denominator_2 = coef_2.bmm(bmm_8)
        bmm_8 = None
        mul_33 = coef_2 * numerator_2
        coef_2 = numerator_2 = None
        add_62 = denominator_2 + 1e-06
        denominator_2 = None
        coef_3 = mul_33 / add_62
        mul_33 = add_62 = None
        numerator_3 = torch.bmm(x_172, coef_3)
        transpose_10 = coef_3.transpose(1, 2)
        bmm_11 = transpose_10.bmm(coef_3)
        transpose_10 = None
        denominator_3 = bases_2.bmm(bmm_11)
        bmm_11 = None
        mul_34 = bases_2 * numerator_3
        bases_2 = numerator_3 = None
        add_63 = denominator_3 + 1e-06
        denominator_3 = None
        bases_3 = mul_34 / add_63
        mul_34 = add_63 = None
        transpose_11 = x_172.transpose(1, 2)
        numerator_4 = torch.bmm(transpose_11, bases_3)
        transpose_11 = None
        transpose_12 = bases_3.transpose(1, 2)
        bmm_14 = transpose_12.bmm(bases_3)
        transpose_12 = None
        denominator_4 = coef_3.bmm(bmm_14)
        bmm_14 = None
        mul_35 = coef_3 * numerator_4
        coef_3 = numerator_4 = None
        add_64 = denominator_4 + 1e-06
        denominator_4 = None
        coef_4 = mul_35 / add_64
        mul_35 = add_64 = None
        numerator_5 = torch.bmm(x_172, coef_4)
        transpose_13 = coef_4.transpose(1, 2)
        bmm_17 = transpose_13.bmm(coef_4)
        transpose_13 = None
        denominator_5 = bases_3.bmm(bmm_17)
        bmm_17 = None
        mul_36 = bases_3 * numerator_5
        bases_3 = numerator_5 = None
        add_65 = denominator_5 + 1e-06
        denominator_5 = None
        bases_4 = mul_36 / add_65
        mul_36 = add_65 = None
        transpose_14 = x_172.transpose(1, 2)
        numerator_6 = torch.bmm(transpose_14, bases_4)
        transpose_14 = None
        transpose_15 = bases_4.transpose(1, 2)
        bmm_20 = transpose_15.bmm(bases_4)
        transpose_15 = None
        denominator_6 = coef_4.bmm(bmm_20)
        bmm_20 = None
        mul_37 = coef_4 * numerator_6
        coef_4 = numerator_6 = None
        add_66 = denominator_6 + 1e-06
        denominator_6 = None
        coef_5 = mul_37 / add_66
        mul_37 = add_66 = None
        numerator_7 = torch.bmm(x_172, coef_5)
        transpose_16 = coef_5.transpose(1, 2)
        bmm_23 = transpose_16.bmm(coef_5)
        transpose_16 = None
        denominator_7 = bases_4.bmm(bmm_23)
        bmm_23 = None
        mul_38 = bases_4 * numerator_7
        bases_4 = numerator_7 = None
        add_67 = denominator_7 + 1e-06
        denominator_7 = None
        bases_5 = mul_38 / add_67
        mul_38 = add_67 = None
        transpose_17 = x_172.transpose(1, 2)
        numerator_8 = torch.bmm(transpose_17, bases_5)
        transpose_17 = None
        transpose_18 = bases_5.transpose(1, 2)
        bmm_26 = transpose_18.bmm(bases_5)
        transpose_18 = None
        denominator_8 = coef_5.bmm(bmm_26)
        bmm_26 = None
        mul_39 = coef_5 * numerator_8
        coef_5 = numerator_8 = None
        add_68 = denominator_8 + 1e-06
        denominator_8 = None
        coef_6 = mul_39 / add_68
        mul_39 = add_68 = None
        numerator_9 = torch.bmm(x_172, coef_6)
        transpose_19 = coef_6.transpose(1, 2)
        bmm_29 = transpose_19.bmm(coef_6)
        transpose_19 = None
        denominator_9 = bases_5.bmm(bmm_29)
        bmm_29 = None
        mul_40 = bases_5 * numerator_9
        bases_5 = numerator_9 = None
        add_69 = denominator_9 + 1e-06
        denominator_9 = None
        bases_6 = mul_40 / add_69
        mul_40 = add_69 = None
        transpose_20 = x_172.transpose(1, 2)
        numerator_10 = torch.bmm(transpose_20, bases_6)
        transpose_20 = None
        transpose_21 = bases_6.transpose(1, 2)
        bmm_32 = transpose_21.bmm(bases_6)
        transpose_21 = None
        denominator_10 = coef_6.bmm(bmm_32)
        bmm_32 = None
        mul_41 = coef_6 * numerator_10
        coef_6 = numerator_10 = None
        add_70 = denominator_10 + 1e-06
        denominator_10 = None
        coef_7 = mul_41 / add_70
        mul_41 = add_70 = None
        numerator_11 = torch.bmm(x_172, coef_7)
        transpose_22 = coef_7.transpose(1, 2)
        bmm_35 = transpose_22.bmm(coef_7)
        transpose_22 = None
        denominator_11 = bases_6.bmm(bmm_35)
        bmm_35 = None
        mul_42 = bases_6 * numerator_11
        bases_6 = numerator_11 = None
        add_71 = denominator_11 + 1e-06
        denominator_11 = None
        bases_7 = mul_42 / add_71
        mul_42 = add_71 = None
        transpose_23 = x_172.transpose(1, 2)
        numerator_12 = torch.bmm(transpose_23, bases_7)
        transpose_23 = None
        transpose_24 = bases_7.transpose(1, 2)
        bmm_38 = transpose_24.bmm(bases_7)
        transpose_24 = None
        denominator_12 = coef_7.bmm(bmm_38)
        bmm_38 = None
        mul_43 = coef_7 * numerator_12
        coef_7 = numerator_12 = None
        add_72 = denominator_12 + 1e-06
        denominator_12 = None
        coef_8 = mul_43 / add_72
        mul_43 = add_72 = None
        numerator_13 = torch.bmm(x_172, coef_8)
        transpose_25 = coef_8.transpose(1, 2)
        bmm_41 = transpose_25.bmm(coef_8)
        transpose_25 = None
        denominator_13 = bases_7.bmm(bmm_41)
        bmm_41 = None
        mul_44 = bases_7 * numerator_13
        bases_7 = numerator_13 = None
        add_73 = denominator_13 + 1e-06
        denominator_13 = None
        bases_8 = mul_44 / add_73
        mul_44 = add_73 = None
        transpose_26 = x_172.transpose(1, 2)
        x_172 = None
        numerator_14 = torch.bmm(transpose_26, bases_8)
        transpose_26 = None
        transpose_27 = bases_8.transpose(1, 2)
        bmm_44 = transpose_27.bmm(bases_8)
        transpose_27 = None
        denominator_14 = coef_8.bmm(bmm_44)
        bmm_44 = None
        mul_45 = coef_8 * numerator_14
        coef_8 = numerator_14 = None
        add_74 = denominator_14 + 1e-06
        denominator_14 = None
        coef_9 = mul_45 / add_74
        mul_45 = add_74 = None
        transpose_28 = coef_9.transpose(1, 2)
        coef_9 = None
        x_173 = torch.bmm(bases_8, transpose_28)
        bases_8 = transpose_28 = None
        x_174 = x_173.view(1, 256, 64, 64)
        x_173 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_conv_parameters_weight_ = (None)
        x_176 = torch.nn.functional.group_norm(
            x_175,
            32,
            l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_gn_parameters_weight_,
            l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_gn_parameters_bias_,
            1e-05,
        )
        x_175 = l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_gn_parameters_weight_ = l_self_modules_decode_head_modules_hamburger_modules_ham_out_modules_gn_parameters_bias_ = (None)
        add_75 = x_170 + x_176
        x_170 = x_176 = None
        ham = torch.nn.functional.relu(add_75, inplace=True)
        add_75 = None
        x_177 = torch.conv2d(
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
        x_178 = torch.nn.functional.group_norm(
            x_177,
            32,
            l_self_modules_decode_head_modules_align_modules_gn_parameters_weight_,
            l_self_modules_decode_head_modules_align_modules_gn_parameters_bias_,
            1e-05,
        )
        x_177 = (
            l_self_modules_decode_head_modules_align_modules_gn_parameters_weight_
        ) = l_self_modules_decode_head_modules_align_modules_gn_parameters_bias_ = None
        x_179 = torch.nn.functional.relu(x_178, inplace=True)
        x_178 = None
        feat = torch.nn.functional.dropout2d(x_179, 0.1, False, False)
        x_179 = None
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
