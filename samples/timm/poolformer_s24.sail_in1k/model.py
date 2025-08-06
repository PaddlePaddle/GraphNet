import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_conv_parameters_bias_ = (
            L_self_modules_stem_modules_conv_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_
        l_self_modules_head_modules_norm_parameters_weight_ = (
            L_self_modules_head_modules_norm_parameters_weight_
        )
        l_self_modules_head_modules_norm_parameters_bias_ = (
            L_self_modules_head_modules_norm_parameters_bias_
        )
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv_parameters_weight_,
            l_self_modules_stem_modules_conv_parameters_bias_,
            (4, 4),
            (2, 2),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_stem_modules_conv_parameters_weight_
        ) = l_self_modules_stem_modules_conv_parameters_bias_ = None
        sym_sum = torch.sym_sum([-3, s1])
        s1 = None
        floordiv = sym_sum // 4
        sym_sum_1 = torch.sym_sum([1, floordiv])
        floordiv = sym_sum_1 = None
        group_norm = torch.nn.functional.group_norm(
            x,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        y = torch._C._nn.avg_pool2d(group_norm, 3, 1, 1, False, False, None)
        sub = y - group_norm
        y = group_norm = None
        view = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_.view(
            (64, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul = sub * view
        sub = view = None
        x_1 = x + mul
        x = mul = None
        group_norm_1 = torch.nn.functional.group_norm(
            x_1,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_2 = torch.conv2d(
            group_norm_1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_1 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_3 = torch._C._nn.gelu(x_2, approximate="none")
        x_2 = None
        x_4 = torch.nn.functional.dropout(x_3, 0.0, False, False)
        x_3 = None
        x_5 = torch.conv2d(
            x_4,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_4 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_6 = torch.nn.functional.dropout(x_5, 0.0, False, False)
        x_5 = None
        view_1 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_.view(
            (64, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_1 = x_6 * view_1
        x_6 = view_1 = None
        x_7 = x_1 + mul_1
        x_1 = mul_1 = None
        group_norm_2 = torch.nn.functional.group_norm(
            x_7,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        y_1 = torch._C._nn.avg_pool2d(group_norm_2, 3, 1, 1, False, False, None)
        sub_1 = y_1 - group_norm_2
        y_1 = group_norm_2 = None
        view_2 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_.view(
            (64, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_2 = sub_1 * view_2
        sub_1 = view_2 = None
        x_8 = x_7 + mul_2
        x_7 = mul_2 = None
        group_norm_3 = torch.nn.functional.group_norm(
            x_8,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_9 = torch.conv2d(
            group_norm_3,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_3 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_10 = torch._C._nn.gelu(x_9, approximate="none")
        x_9 = None
        x_11 = torch.nn.functional.dropout(x_10, 0.0, False, False)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_13 = torch.nn.functional.dropout(x_12, 0.0, False, False)
        x_12 = None
        view_3 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_.view(
            (64, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_3 = x_13 * view_3
        x_13 = view_3 = None
        x_14 = x_8 + mul_3
        x_8 = mul_3 = None
        group_norm_4 = torch.nn.functional.group_norm(
            x_14,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        y_2 = torch._C._nn.avg_pool2d(group_norm_4, 3, 1, 1, False, False, None)
        sub_2 = y_2 - group_norm_4
        y_2 = group_norm_4 = None
        view_4 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_.view(
            (64, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_4 = sub_2 * view_4
        sub_2 = view_4 = None
        x_15 = x_14 + mul_4
        x_14 = mul_4 = None
        group_norm_5 = torch.nn.functional.group_norm(
            x_15,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_16 = torch.conv2d(
            group_norm_5,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_5 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_17 = torch._C._nn.gelu(x_16, approximate="none")
        x_16 = None
        x_18 = torch.nn.functional.dropout(x_17, 0.0, False, False)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_20 = torch.nn.functional.dropout(x_19, 0.0, False, False)
        x_19 = None
        view_5 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_.view(
            (64, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_5 = x_20 * view_5
        x_20 = view_5 = None
        x_21 = x_15 + mul_5
        x_15 = mul_5 = None
        group_norm_6 = torch.nn.functional.group_norm(
            x_21,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        y_3 = torch._C._nn.avg_pool2d(group_norm_6, 3, 1, 1, False, False, None)
        sub_3 = y_3 - group_norm_6
        y_3 = group_norm_6 = None
        view_6 = l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_.view(
            (64, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_6 = sub_3 * view_6
        sub_3 = view_6 = None
        x_22 = x_21 + mul_6
        x_21 = mul_6 = None
        group_norm_7 = torch.nn.functional.group_norm(
            x_22,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_23 = torch.conv2d(
            group_norm_7,
            l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_7 = l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_24 = torch._C._nn.gelu(x_23, approximate="none")
        x_23 = None
        x_25 = torch.nn.functional.dropout(x_24, 0.0, False, False)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_27 = torch.nn.functional.dropout(x_26, 0.0, False, False)
        x_26 = None
        view_7 = l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_.view(
            (64, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_7 = x_27 * view_7
        x_27 = view_7 = None
        x_28 = x_22 + mul_7
        x_22 = mul_7 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = (None)
        floordiv_1 = sym_sum // 8
        sym_sum_2 = torch.sym_sum([1, floordiv_1])
        floordiv_1 = sym_sum_2 = None
        group_norm_8 = torch.nn.functional.group_norm(
            x_29,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        y_4 = torch._C._nn.avg_pool2d(group_norm_8, 3, 1, 1, False, False, None)
        sub_4 = y_4 - group_norm_8
        y_4 = group_norm_8 = None
        view_8 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_.view(
            (128, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_8 = sub_4 * view_8
        sub_4 = view_8 = None
        x_30 = x_29 + mul_8
        x_29 = mul_8 = None
        group_norm_9 = torch.nn.functional.group_norm(
            x_30,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_31 = torch.conv2d(
            group_norm_9,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_9 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_32 = torch._C._nn.gelu(x_31, approximate="none")
        x_31 = None
        x_33 = torch.nn.functional.dropout(x_32, 0.0, False, False)
        x_32 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_35 = torch.nn.functional.dropout(x_34, 0.0, False, False)
        x_34 = None
        view_9 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_.view(
            (128, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_9 = x_35 * view_9
        x_35 = view_9 = None
        x_36 = x_30 + mul_9
        x_30 = mul_9 = None
        group_norm_10 = torch.nn.functional.group_norm(
            x_36,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        y_5 = torch._C._nn.avg_pool2d(group_norm_10, 3, 1, 1, False, False, None)
        sub_5 = y_5 - group_norm_10
        y_5 = group_norm_10 = None
        view_10 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_.view(
            (128, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_10 = sub_5 * view_10
        sub_5 = view_10 = None
        x_37 = x_36 + mul_10
        x_36 = mul_10 = None
        group_norm_11 = torch.nn.functional.group_norm(
            x_37,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_38 = torch.conv2d(
            group_norm_11,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_11 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_39 = torch._C._nn.gelu(x_38, approximate="none")
        x_38 = None
        x_40 = torch.nn.functional.dropout(x_39, 0.0, False, False)
        x_39 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_42 = torch.nn.functional.dropout(x_41, 0.0, False, False)
        x_41 = None
        view_11 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_.view(
            (128, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_11 = x_42 * view_11
        x_42 = view_11 = None
        x_43 = x_37 + mul_11
        x_37 = mul_11 = None
        group_norm_12 = torch.nn.functional.group_norm(
            x_43,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        y_6 = torch._C._nn.avg_pool2d(group_norm_12, 3, 1, 1, False, False, None)
        sub_6 = y_6 - group_norm_12
        y_6 = group_norm_12 = None
        view_12 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_.view(
            (128, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_12 = sub_6 * view_12
        sub_6 = view_12 = None
        x_44 = x_43 + mul_12
        x_43 = mul_12 = None
        group_norm_13 = torch.nn.functional.group_norm(
            x_44,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_45 = torch.conv2d(
            group_norm_13,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_13 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_46 = torch._C._nn.gelu(x_45, approximate="none")
        x_45 = None
        x_47 = torch.nn.functional.dropout(x_46, 0.0, False, False)
        x_46 = None
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_47 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_49 = torch.nn.functional.dropout(x_48, 0.0, False, False)
        x_48 = None
        view_13 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_.view(
            (128, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_13 = x_49 * view_13
        x_49 = view_13 = None
        x_50 = x_44 + mul_13
        x_44 = mul_13 = None
        group_norm_14 = torch.nn.functional.group_norm(
            x_50,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        y_7 = torch._C._nn.avg_pool2d(group_norm_14, 3, 1, 1, False, False, None)
        sub_7 = y_7 - group_norm_14
        y_7 = group_norm_14 = None
        view_14 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_.view(
            (128, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_14 = sub_7 * view_14
        sub_7 = view_14 = None
        x_51 = x_50 + mul_14
        x_50 = mul_14 = None
        group_norm_15 = torch.nn.functional.group_norm(
            x_51,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_52 = torch.conv2d(
            group_norm_15,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_15 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_53 = torch._C._nn.gelu(x_52, approximate="none")
        x_52 = None
        x_54 = torch.nn.functional.dropout(x_53, 0.0, False, False)
        x_53 = None
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_54 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_56 = torch.nn.functional.dropout(x_55, 0.0, False, False)
        x_55 = None
        view_15 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_.view(
            (128, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_15 = x_56 * view_15
        x_56 = view_15 = None
        x_57 = x_51 + mul_15
        x_51 = mul_15 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = (None)
        floordiv_2 = sym_sum // 16
        sym_sum_3 = torch.sym_sum([1, floordiv_2])
        floordiv_2 = sym_sum_3 = None
        group_norm_16 = torch.nn.functional.group_norm(
            x_58,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        y_8 = torch._C._nn.avg_pool2d(group_norm_16, 3, 1, 1, False, False, None)
        sub_8 = y_8 - group_norm_16
        y_8 = group_norm_16 = None
        view_16 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_16 = sub_8 * view_16
        sub_8 = view_16 = None
        x_59 = x_58 + mul_16
        x_58 = mul_16 = None
        group_norm_17 = torch.nn.functional.group_norm(
            x_59,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_60 = torch.conv2d(
            group_norm_17,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_17 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_61 = torch._C._nn.gelu(x_60, approximate="none")
        x_60 = None
        x_62 = torch.nn.functional.dropout(x_61, 0.0, False, False)
        x_61 = None
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_62 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_64 = torch.nn.functional.dropout(x_63, 0.0, False, False)
        x_63 = None
        view_17 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_17 = x_64 * view_17
        x_64 = view_17 = None
        x_65 = x_59 + mul_17
        x_59 = mul_17 = None
        group_norm_18 = torch.nn.functional.group_norm(
            x_65,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        y_9 = torch._C._nn.avg_pool2d(group_norm_18, 3, 1, 1, False, False, None)
        sub_9 = y_9 - group_norm_18
        y_9 = group_norm_18 = None
        view_18 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_18 = sub_9 * view_18
        sub_9 = view_18 = None
        x_66 = x_65 + mul_18
        x_65 = mul_18 = None
        group_norm_19 = torch.nn.functional.group_norm(
            x_66,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_67 = torch.conv2d(
            group_norm_19,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_19 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_68 = torch._C._nn.gelu(x_67, approximate="none")
        x_67 = None
        x_69 = torch.nn.functional.dropout(x_68, 0.0, False, False)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_71 = torch.nn.functional.dropout(x_70, 0.0, False, False)
        x_70 = None
        view_19 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_19 = x_71 * view_19
        x_71 = view_19 = None
        x_72 = x_66 + mul_19
        x_66 = mul_19 = None
        group_norm_20 = torch.nn.functional.group_norm(
            x_72,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        y_10 = torch._C._nn.avg_pool2d(group_norm_20, 3, 1, 1, False, False, None)
        sub_10 = y_10 - group_norm_20
        y_10 = group_norm_20 = None
        view_20 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_20 = sub_10 * view_20
        sub_10 = view_20 = None
        x_73 = x_72 + mul_20
        x_72 = mul_20 = None
        group_norm_21 = torch.nn.functional.group_norm(
            x_73,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_74 = torch.conv2d(
            group_norm_21,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_21 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_75 = torch._C._nn.gelu(x_74, approximate="none")
        x_74 = None
        x_76 = torch.nn.functional.dropout(x_75, 0.0, False, False)
        x_75 = None
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_76 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_78 = torch.nn.functional.dropout(x_77, 0.0, False, False)
        x_77 = None
        view_21 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_21 = x_78 * view_21
        x_78 = view_21 = None
        x_79 = x_73 + mul_21
        x_73 = mul_21 = None
        group_norm_22 = torch.nn.functional.group_norm(
            x_79,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        y_11 = torch._C._nn.avg_pool2d(group_norm_22, 3, 1, 1, False, False, None)
        sub_11 = y_11 - group_norm_22
        y_11 = group_norm_22 = None
        view_22 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_22 = sub_11 * view_22
        sub_11 = view_22 = None
        x_80 = x_79 + mul_22
        x_79 = mul_22 = None
        group_norm_23 = torch.nn.functional.group_norm(
            x_80,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_81 = torch.conv2d(
            group_norm_23,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_23 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_82 = torch._C._nn.gelu(x_81, approximate="none")
        x_81 = None
        x_83 = torch.nn.functional.dropout(x_82, 0.0, False, False)
        x_82 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_83 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_85 = torch.nn.functional.dropout(x_84, 0.0, False, False)
        x_84 = None
        view_23 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_23 = x_85 * view_23
        x_85 = view_23 = None
        x_86 = x_80 + mul_23
        x_80 = mul_23 = None
        group_norm_24 = torch.nn.functional.group_norm(
            x_86,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        y_12 = torch._C._nn.avg_pool2d(group_norm_24, 3, 1, 1, False, False, None)
        sub_12 = y_12 - group_norm_24
        y_12 = group_norm_24 = None
        view_24 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_24 = sub_12 * view_24
        sub_12 = view_24 = None
        x_87 = x_86 + mul_24
        x_86 = mul_24 = None
        group_norm_25 = torch.nn.functional.group_norm(
            x_87,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_88 = torch.conv2d(
            group_norm_25,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_25 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_89 = torch._C._nn.gelu(x_88, approximate="none")
        x_88 = None
        x_90 = torch.nn.functional.dropout(x_89, 0.0, False, False)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_90 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_92 = torch.nn.functional.dropout(x_91, 0.0, False, False)
        x_91 = None
        view_25 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_25 = x_92 * view_25
        x_92 = view_25 = None
        x_93 = x_87 + mul_25
        x_87 = mul_25 = None
        group_norm_26 = torch.nn.functional.group_norm(
            x_93,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        y_13 = torch._C._nn.avg_pool2d(group_norm_26, 3, 1, 1, False, False, None)
        sub_13 = y_13 - group_norm_26
        y_13 = group_norm_26 = None
        view_26 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_26 = sub_13 * view_26
        sub_13 = view_26 = None
        x_94 = x_93 + mul_26
        x_93 = mul_26 = None
        group_norm_27 = torch.nn.functional.group_norm(
            x_94,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_95 = torch.conv2d(
            group_norm_27,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_27 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_96 = torch._C._nn.gelu(x_95, approximate="none")
        x_95 = None
        x_97 = torch.nn.functional.dropout(x_96, 0.0, False, False)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_99 = torch.nn.functional.dropout(x_98, 0.0, False, False)
        x_98 = None
        view_27 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_27 = x_99 * view_27
        x_99 = view_27 = None
        x_100 = x_94 + mul_27
        x_94 = mul_27 = None
        group_norm_28 = torch.nn.functional.group_norm(
            x_100,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (None)
        y_14 = torch._C._nn.avg_pool2d(group_norm_28, 3, 1, 1, False, False, None)
        sub_14 = y_14 - group_norm_28
        y_14 = group_norm_28 = None
        view_28 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_28 = sub_14 * view_28
        sub_14 = view_28 = None
        x_101 = x_100 + mul_28
        x_100 = mul_28 = None
        group_norm_29 = torch.nn.functional.group_norm(
            x_101,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (None)
        x_102 = torch.conv2d(
            group_norm_29,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_29 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_103 = torch._C._nn.gelu(x_102, approximate="none")
        x_102 = None
        x_104 = torch.nn.functional.dropout(x_103, 0.0, False, False)
        x_103 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        view_29 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_29 = x_106 * view_29
        x_106 = view_29 = None
        x_107 = x_101 + mul_29
        x_101 = mul_29 = None
        group_norm_30 = torch.nn.functional.group_norm(
            x_107,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (None)
        y_15 = torch._C._nn.avg_pool2d(group_norm_30, 3, 1, 1, False, False, None)
        sub_15 = y_15 - group_norm_30
        y_15 = group_norm_30 = None
        view_30 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_30 = sub_15 * view_30
        sub_15 = view_30 = None
        x_108 = x_107 + mul_30
        x_107 = mul_30 = None
        group_norm_31 = torch.nn.functional.group_norm(
            x_108,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (None)
        x_109 = torch.conv2d(
            group_norm_31,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_31 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_110 = torch._C._nn.gelu(x_109, approximate="none")
        x_109 = None
        x_111 = torch.nn.functional.dropout(x_110, 0.0, False, False)
        x_110 = None
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_111 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_113 = torch.nn.functional.dropout(x_112, 0.0, False, False)
        x_112 = None
        view_31 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_31 = x_113 * view_31
        x_113 = view_31 = None
        x_114 = x_108 + mul_31
        x_108 = mul_31 = None
        group_norm_32 = torch.nn.functional.group_norm(
            x_114,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (None)
        y_16 = torch._C._nn.avg_pool2d(group_norm_32, 3, 1, 1, False, False, None)
        sub_16 = y_16 - group_norm_32
        y_16 = group_norm_32 = None
        view_32 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_layer_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_32 = sub_16 * view_32
        sub_16 = view_32 = None
        x_115 = x_114 + mul_32
        x_114 = mul_32 = None
        group_norm_33 = torch.nn.functional.group_norm(
            x_115,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (None)
        x_116 = torch.conv2d(
            group_norm_33,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_33 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_117 = torch._C._nn.gelu(x_116, approximate="none")
        x_116 = None
        x_118 = torch.nn.functional.dropout(x_117, 0.0, False, False)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        view_33 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_layer_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_33 = x_120 * view_33
        x_120 = view_33 = None
        x_121 = x_115 + mul_33
        x_115 = mul_33 = None
        group_norm_34 = torch.nn.functional.group_norm(
            x_121,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (None)
        y_17 = torch._C._nn.avg_pool2d(group_norm_34, 3, 1, 1, False, False, None)
        sub_17 = y_17 - group_norm_34
        y_17 = group_norm_34 = None
        view_34 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_layer_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_34 = sub_17 * view_34
        sub_17 = view_34 = None
        x_122 = x_121 + mul_34
        x_121 = mul_34 = None
        group_norm_35 = torch.nn.functional.group_norm(
            x_122,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (None)
        x_123 = torch.conv2d(
            group_norm_35,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_35 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_124 = torch._C._nn.gelu(x_123, approximate="none")
        x_123 = None
        x_125 = torch.nn.functional.dropout(x_124, 0.0, False, False)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_127 = torch.nn.functional.dropout(x_126, 0.0, False, False)
        x_126 = None
        view_35 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_layer_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_35 = x_127 * view_35
        x_127 = view_35 = None
        x_128 = x_122 + mul_35
        x_122 = mul_35 = None
        group_norm_36 = torch.nn.functional.group_norm(
            x_128,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (None)
        y_18 = torch._C._nn.avg_pool2d(group_norm_36, 3, 1, 1, False, False, None)
        sub_18 = y_18 - group_norm_36
        y_18 = group_norm_36 = None
        view_36 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_layer_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_36 = sub_18 * view_36
        sub_18 = view_36 = None
        x_129 = x_128 + mul_36
        x_128 = mul_36 = None
        group_norm_37 = torch.nn.functional.group_norm(
            x_129,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (None)
        x_130 = torch.conv2d(
            group_norm_37,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_37 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_131 = torch._C._nn.gelu(x_130, approximate="none")
        x_130 = None
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_132 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_134 = torch.nn.functional.dropout(x_133, 0.0, False, False)
        x_133 = None
        view_37 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_layer_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_37 = x_134 * view_37
        x_134 = view_37 = None
        x_135 = x_129 + mul_37
        x_129 = mul_37 = None
        group_norm_38 = torch.nn.functional.group_norm(
            x_135,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (None)
        y_19 = torch._C._nn.avg_pool2d(group_norm_38, 3, 1, 1, False, False, None)
        sub_19 = y_19 - group_norm_38
        y_19 = group_norm_38 = None
        view_38 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_layer_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_38 = sub_19 * view_38
        sub_19 = view_38 = None
        x_136 = x_135 + mul_38
        x_135 = mul_38 = None
        group_norm_39 = torch.nn.functional.group_norm(
            x_136,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (None)
        x_137 = torch.conv2d(
            group_norm_39,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_39 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_138 = torch._C._nn.gelu(x_137, approximate="none")
        x_137 = None
        x_139 = torch.nn.functional.dropout(x_138, 0.0, False, False)
        x_138 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_141 = torch.nn.functional.dropout(x_140, 0.0, False, False)
        x_140 = None
        view_39 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_layer_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_39 = x_141 * view_39
        x_141 = view_39 = None
        x_142 = x_136 + mul_39
        x_136 = mul_39 = None
        x_143 = torch.conv2d(
            x_142,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_142 = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = (None)
        floordiv_3 = sym_sum // 32
        sym_sum = None
        sym_sum_4 = torch.sym_sum([1, floordiv_3])
        floordiv_3 = sym_sum_4 = None
        group_norm_40 = torch.nn.functional.group_norm(
            x_143,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        y_20 = torch._C._nn.avg_pool2d(group_norm_40, 3, 1, 1, False, False, None)
        sub_20 = y_20 - group_norm_40
        y_20 = group_norm_40 = None
        view_40 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_40 = sub_20 * view_40
        sub_20 = view_40 = None
        x_144 = x_143 + mul_40
        x_143 = mul_40 = None
        group_norm_41 = torch.nn.functional.group_norm(
            x_144,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_145 = torch.conv2d(
            group_norm_41,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_41 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_146 = torch._C._nn.gelu(x_145, approximate="none")
        x_145 = None
        x_147 = torch.nn.functional.dropout(x_146, 0.0, False, False)
        x_146 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_147 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_149 = torch.nn.functional.dropout(x_148, 0.0, False, False)
        x_148 = None
        view_41 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_41 = x_149 * view_41
        x_149 = view_41 = None
        x_150 = x_144 + mul_41
        x_144 = mul_41 = None
        group_norm_42 = torch.nn.functional.group_norm(
            x_150,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        y_21 = torch._C._nn.avg_pool2d(group_norm_42, 3, 1, 1, False, False, None)
        sub_21 = y_21 - group_norm_42
        y_21 = group_norm_42 = None
        view_42 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_42 = sub_21 * view_42
        sub_21 = view_42 = None
        x_151 = x_150 + mul_42
        x_150 = mul_42 = None
        group_norm_43 = torch.nn.functional.group_norm(
            x_151,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_152 = torch.conv2d(
            group_norm_43,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_43 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_153 = torch._C._nn.gelu(x_152, approximate="none")
        x_152 = None
        x_154 = torch.nn.functional.dropout(x_153, 0.0, False, False)
        x_153 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_156 = torch.nn.functional.dropout(x_155, 0.0, False, False)
        x_155 = None
        view_43 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_43 = x_156 * view_43
        x_156 = view_43 = None
        x_157 = x_151 + mul_43
        x_151 = mul_43 = None
        group_norm_44 = torch.nn.functional.group_norm(
            x_157,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        y_22 = torch._C._nn.avg_pool2d(group_norm_44, 3, 1, 1, False, False, None)
        sub_22 = y_22 - group_norm_44
        y_22 = group_norm_44 = None
        view_44 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_44 = sub_22 * view_44
        sub_22 = view_44 = None
        x_158 = x_157 + mul_44
        x_157 = mul_44 = None
        group_norm_45 = torch.nn.functional.group_norm(
            x_158,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_159 = torch.conv2d(
            group_norm_45,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_45 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_160 = torch._C._nn.gelu(x_159, approximate="none")
        x_159 = None
        x_161 = torch.nn.functional.dropout(x_160, 0.0, False, False)
        x_160 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_163 = torch.nn.functional.dropout(x_162, 0.0, False, False)
        x_162 = None
        view_45 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_45 = x_163 * view_45
        x_163 = view_45 = None
        x_164 = x_158 + mul_45
        x_158 = mul_45 = None
        group_norm_46 = torch.nn.functional.group_norm(
            x_164,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        y_23 = torch._C._nn.avg_pool2d(group_norm_46, 3, 1, 1, False, False, None)
        sub_23 = y_23 - group_norm_46
        y_23 = group_norm_46 = None
        view_46 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_46 = sub_23 * view_46
        sub_23 = view_46 = None
        x_165 = x_164 + mul_46
        x_164 = mul_46 = None
        group_norm_47 = torch.nn.functional.group_norm(
            x_165,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_166 = torch.conv2d(
            group_norm_47,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_47 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_167 = torch._C._nn.gelu(x_166, approximate="none")
        x_166 = None
        x_168 = torch.nn.functional.dropout(x_167, 0.0, False, False)
        x_167 = None
        x_169 = torch.conv2d(
            x_168,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_168 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_170 = torch.nn.functional.dropout(x_169, 0.0, False, False)
        x_169 = None
        view_47 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_47 = x_170 * view_47
        x_170 = view_47 = None
        x_171 = x_165 + mul_47
        x_165 = mul_47 = None
        x_172 = torch.nn.functional.adaptive_avg_pool2d(x_171, 1)
        x_171 = None
        x_173 = x_172.permute(0, 2, 3, 1)
        x_172 = None
        x_174 = torch.nn.functional.layer_norm(
            x_173,
            (512,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_173 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_175 = x_174.permute(0, 3, 1, 2)
        x_174 = None
        x_176 = x_175.flatten(1, -1)
        x_175 = None
        x_177 = torch._C._nn.linear(
            x_176,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_176 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_177,)
