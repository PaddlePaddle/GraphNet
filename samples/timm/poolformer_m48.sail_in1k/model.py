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
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_
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
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_layer_scale2_parameters_scale_
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
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_
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
            (96, 1, 1)
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
            (96, 1, 1)
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
            (96, 1, 1)
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
            (96, 1, 1)
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
            (96, 1, 1)
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
            (96, 1, 1)
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
            (96, 1, 1)
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
            (96, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_7 = x_27 * view_7
        x_27 = view_7 = None
        x_28 = x_22 + mul_7
        x_22 = mul_7 = None
        group_norm_8 = torch.nn.functional.group_norm(
            x_28,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        y_4 = torch._C._nn.avg_pool2d(group_norm_8, 3, 1, 1, False, False, None)
        sub_4 = y_4 - group_norm_8
        y_4 = group_norm_8 = None
        view_8 = l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_.view(
            (96, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_8 = sub_4 * view_8
        sub_4 = view_8 = None
        x_29 = x_28 + mul_8
        x_28 = mul_8 = None
        group_norm_9 = torch.nn.functional.group_norm(
            x_29,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_30 = torch.conv2d(
            group_norm_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_9 = l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_31 = torch._C._nn.gelu(x_30, approximate="none")
        x_30 = None
        x_32 = torch.nn.functional.dropout(x_31, 0.0, False, False)
        x_31 = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_34 = torch.nn.functional.dropout(x_33, 0.0, False, False)
        x_33 = None
        view_9 = l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_.view(
            (96, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_9 = x_34 * view_9
        x_34 = view_9 = None
        x_35 = x_29 + mul_9
        x_29 = mul_9 = None
        group_norm_10 = torch.nn.functional.group_norm(
            x_35,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        y_5 = torch._C._nn.avg_pool2d(group_norm_10, 3, 1, 1, False, False, None)
        sub_5 = y_5 - group_norm_10
        y_5 = group_norm_10 = None
        view_10 = l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_.view(
            (96, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_10 = sub_5 * view_10
        sub_5 = view_10 = None
        x_36 = x_35 + mul_10
        x_35 = mul_10 = None
        group_norm_11 = torch.nn.functional.group_norm(
            x_36,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_37 = torch.conv2d(
            group_norm_11,
            l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_11 = l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_38 = torch._C._nn.gelu(x_37, approximate="none")
        x_37 = None
        x_39 = torch.nn.functional.dropout(x_38, 0.0, False, False)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_41 = torch.nn.functional.dropout(x_40, 0.0, False, False)
        x_40 = None
        view_11 = l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_.view(
            (96, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_11 = x_41 * view_11
        x_41 = view_11 = None
        x_42 = x_36 + mul_11
        x_36 = mul_11 = None
        group_norm_12 = torch.nn.functional.group_norm(
            x_42,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (None)
        y_6 = torch._C._nn.avg_pool2d(group_norm_12, 3, 1, 1, False, False, None)
        sub_6 = y_6 - group_norm_12
        y_6 = group_norm_12 = None
        view_12 = l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_.view(
            (96, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_12 = sub_6 * view_12
        sub_6 = view_12 = None
        x_43 = x_42 + mul_12
        x_42 = mul_12 = None
        group_norm_13 = torch.nn.functional.group_norm(
            x_43,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (None)
        x_44 = torch.conv2d(
            group_norm_13,
            l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_13 = l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_45 = torch._C._nn.gelu(x_44, approximate="none")
        x_44 = None
        x_46 = torch.nn.functional.dropout(x_45, 0.0, False, False)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_48 = torch.nn.functional.dropout(x_47, 0.0, False, False)
        x_47 = None
        view_13 = l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_.view(
            (96, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_13 = x_48 * view_13
        x_48 = view_13 = None
        x_49 = x_43 + mul_13
        x_43 = mul_13 = None
        group_norm_14 = torch.nn.functional.group_norm(
            x_49,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (None)
        y_7 = torch._C._nn.avg_pool2d(group_norm_14, 3, 1, 1, False, False, None)
        sub_7 = y_7 - group_norm_14
        y_7 = group_norm_14 = None
        view_14 = l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_.view(
            (96, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_14 = sub_7 * view_14
        sub_7 = view_14 = None
        x_50 = x_49 + mul_14
        x_49 = mul_14 = None
        group_norm_15 = torch.nn.functional.group_norm(
            x_50,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (None)
        x_51 = torch.conv2d(
            group_norm_15,
            l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_15 = l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_52 = torch._C._nn.gelu(x_51, approximate="none")
        x_51 = None
        x_53 = torch.nn.functional.dropout(x_52, 0.0, False, False)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_55 = torch.nn.functional.dropout(x_54, 0.0, False, False)
        x_54 = None
        view_15 = l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_.view(
            (96, 1, 1)
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_15 = x_55 * view_15
        x_55 = view_15 = None
        x_56 = x_50 + mul_15
        x_50 = mul_15 = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_56 = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = (None)
        floordiv_1 = sym_sum // 8
        sym_sum_2 = torch.sym_sum([1, floordiv_1])
        floordiv_1 = sym_sum_2 = None
        group_norm_16 = torch.nn.functional.group_norm(
            x_57,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        y_8 = torch._C._nn.avg_pool2d(group_norm_16, 3, 1, 1, False, False, None)
        sub_8 = y_8 - group_norm_16
        y_8 = group_norm_16 = None
        view_16 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_16 = sub_8 * view_16
        sub_8 = view_16 = None
        x_58 = x_57 + mul_16
        x_57 = mul_16 = None
        group_norm_17 = torch.nn.functional.group_norm(
            x_58,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_59 = torch.conv2d(
            group_norm_17,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_17 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_60 = torch._C._nn.gelu(x_59, approximate="none")
        x_59 = None
        x_61 = torch.nn.functional.dropout(x_60, 0.0, False, False)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_63 = torch.nn.functional.dropout(x_62, 0.0, False, False)
        x_62 = None
        view_17 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_17 = x_63 * view_17
        x_63 = view_17 = None
        x_64 = x_58 + mul_17
        x_58 = mul_17 = None
        group_norm_18 = torch.nn.functional.group_norm(
            x_64,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        y_9 = torch._C._nn.avg_pool2d(group_norm_18, 3, 1, 1, False, False, None)
        sub_9 = y_9 - group_norm_18
        y_9 = group_norm_18 = None
        view_18 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_18 = sub_9 * view_18
        sub_9 = view_18 = None
        x_65 = x_64 + mul_18
        x_64 = mul_18 = None
        group_norm_19 = torch.nn.functional.group_norm(
            x_65,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_66 = torch.conv2d(
            group_norm_19,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_19 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_67 = torch._C._nn.gelu(x_66, approximate="none")
        x_66 = None
        x_68 = torch.nn.functional.dropout(x_67, 0.0, False, False)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_70 = torch.nn.functional.dropout(x_69, 0.0, False, False)
        x_69 = None
        view_19 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_19 = x_70 * view_19
        x_70 = view_19 = None
        x_71 = x_65 + mul_19
        x_65 = mul_19 = None
        group_norm_20 = torch.nn.functional.group_norm(
            x_71,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        y_10 = torch._C._nn.avg_pool2d(group_norm_20, 3, 1, 1, False, False, None)
        sub_10 = y_10 - group_norm_20
        y_10 = group_norm_20 = None
        view_20 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_20 = sub_10 * view_20
        sub_10 = view_20 = None
        x_72 = x_71 + mul_20
        x_71 = mul_20 = None
        group_norm_21 = torch.nn.functional.group_norm(
            x_72,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_73 = torch.conv2d(
            group_norm_21,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_21 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_74 = torch._C._nn.gelu(x_73, approximate="none")
        x_73 = None
        x_75 = torch.nn.functional.dropout(x_74, 0.0, False, False)
        x_74 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_77 = torch.nn.functional.dropout(x_76, 0.0, False, False)
        x_76 = None
        view_21 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_21 = x_77 * view_21
        x_77 = view_21 = None
        x_78 = x_72 + mul_21
        x_72 = mul_21 = None
        group_norm_22 = torch.nn.functional.group_norm(
            x_78,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        y_11 = torch._C._nn.avg_pool2d(group_norm_22, 3, 1, 1, False, False, None)
        sub_11 = y_11 - group_norm_22
        y_11 = group_norm_22 = None
        view_22 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_22 = sub_11 * view_22
        sub_11 = view_22 = None
        x_79 = x_78 + mul_22
        x_78 = mul_22 = None
        group_norm_23 = torch.nn.functional.group_norm(
            x_79,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_80 = torch.conv2d(
            group_norm_23,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_23 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_81 = torch._C._nn.gelu(x_80, approximate="none")
        x_80 = None
        x_82 = torch.nn.functional.dropout(x_81, 0.0, False, False)
        x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_84 = torch.nn.functional.dropout(x_83, 0.0, False, False)
        x_83 = None
        view_23 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_23 = x_84 * view_23
        x_84 = view_23 = None
        x_85 = x_79 + mul_23
        x_79 = mul_23 = None
        group_norm_24 = torch.nn.functional.group_norm(
            x_85,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        y_12 = torch._C._nn.avg_pool2d(group_norm_24, 3, 1, 1, False, False, None)
        sub_12 = y_12 - group_norm_24
        y_12 = group_norm_24 = None
        view_24 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_24 = sub_12 * view_24
        sub_12 = view_24 = None
        x_86 = x_85 + mul_24
        x_85 = mul_24 = None
        group_norm_25 = torch.nn.functional.group_norm(
            x_86,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_87 = torch.conv2d(
            group_norm_25,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_25 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_88 = torch._C._nn.gelu(x_87, approximate="none")
        x_87 = None
        x_89 = torch.nn.functional.dropout(x_88, 0.0, False, False)
        x_88 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        view_25 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_25 = x_91 * view_25
        x_91 = view_25 = None
        x_92 = x_86 + mul_25
        x_86 = mul_25 = None
        group_norm_26 = torch.nn.functional.group_norm(
            x_92,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        y_13 = torch._C._nn.avg_pool2d(group_norm_26, 3, 1, 1, False, False, None)
        sub_13 = y_13 - group_norm_26
        y_13 = group_norm_26 = None
        view_26 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_26 = sub_13 * view_26
        sub_13 = view_26 = None
        x_93 = x_92 + mul_26
        x_92 = mul_26 = None
        group_norm_27 = torch.nn.functional.group_norm(
            x_93,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_94 = torch.conv2d(
            group_norm_27,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_27 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_95 = torch._C._nn.gelu(x_94, approximate="none")
        x_94 = None
        x_96 = torch.nn.functional.dropout(x_95, 0.0, False, False)
        x_95 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_98 = torch.nn.functional.dropout(x_97, 0.0, False, False)
        x_97 = None
        view_27 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_27 = x_98 * view_27
        x_98 = view_27 = None
        x_99 = x_93 + mul_27
        x_93 = mul_27 = None
        group_norm_28 = torch.nn.functional.group_norm(
            x_99,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (None)
        y_14 = torch._C._nn.avg_pool2d(group_norm_28, 3, 1, 1, False, False, None)
        sub_14 = y_14 - group_norm_28
        y_14 = group_norm_28 = None
        view_28 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_28 = sub_14 * view_28
        sub_14 = view_28 = None
        x_100 = x_99 + mul_28
        x_99 = mul_28 = None
        group_norm_29 = torch.nn.functional.group_norm(
            x_100,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (None)
        x_101 = torch.conv2d(
            group_norm_29,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_29 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_102 = torch._C._nn.gelu(x_101, approximate="none")
        x_101 = None
        x_103 = torch.nn.functional.dropout(x_102, 0.0, False, False)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_105 = torch.nn.functional.dropout(x_104, 0.0, False, False)
        x_104 = None
        view_29 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_29 = x_105 * view_29
        x_105 = view_29 = None
        x_106 = x_100 + mul_29
        x_100 = mul_29 = None
        group_norm_30 = torch.nn.functional.group_norm(
            x_106,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (None)
        y_15 = torch._C._nn.avg_pool2d(group_norm_30, 3, 1, 1, False, False, None)
        sub_15 = y_15 - group_norm_30
        y_15 = group_norm_30 = None
        view_30 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_30 = sub_15 * view_30
        sub_15 = view_30 = None
        x_107 = x_106 + mul_30
        x_106 = mul_30 = None
        group_norm_31 = torch.nn.functional.group_norm(
            x_107,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (None)
        x_108 = torch.conv2d(
            group_norm_31,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_31 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_109 = torch._C._nn.gelu(x_108, approximate="none")
        x_108 = None
        x_110 = torch.nn.functional.dropout(x_109, 0.0, False, False)
        x_109 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_110 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_112 = torch.nn.functional.dropout(x_111, 0.0, False, False)
        x_111 = None
        view_31 = l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_.view(
            (192, 1, 1)
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_31 = x_112 * view_31
        x_112 = view_31 = None
        x_113 = x_107 + mul_31
        x_107 = mul_31 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_113 = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = (None)
        floordiv_2 = sym_sum // 16
        sym_sum_3 = torch.sym_sum([1, floordiv_2])
        floordiv_2 = sym_sum_3 = None
        group_norm_32 = torch.nn.functional.group_norm(
            x_114,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        y_16 = torch._C._nn.avg_pool2d(group_norm_32, 3, 1, 1, False, False, None)
        sub_16 = y_16 - group_norm_32
        y_16 = group_norm_32 = None
        view_32 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_32 = sub_16 * view_32
        sub_16 = view_32 = None
        x_115 = x_114 + mul_32
        x_114 = mul_32 = None
        group_norm_33 = torch.nn.functional.group_norm(
            x_115,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_116 = torch.conv2d(
            group_norm_33,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_33 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_117 = torch._C._nn.gelu(x_116, approximate="none")
        x_116 = None
        x_118 = torch.nn.functional.dropout(x_117, 0.0, False, False)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        view_33 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_33 = x_120 * view_33
        x_120 = view_33 = None
        x_121 = x_115 + mul_33
        x_115 = mul_33 = None
        group_norm_34 = torch.nn.functional.group_norm(
            x_121,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        y_17 = torch._C._nn.avg_pool2d(group_norm_34, 3, 1, 1, False, False, None)
        sub_17 = y_17 - group_norm_34
        y_17 = group_norm_34 = None
        view_34 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_34 = sub_17 * view_34
        sub_17 = view_34 = None
        x_122 = x_121 + mul_34
        x_121 = mul_34 = None
        group_norm_35 = torch.nn.functional.group_norm(
            x_122,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_123 = torch.conv2d(
            group_norm_35,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_35 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_124 = torch._C._nn.gelu(x_123, approximate="none")
        x_123 = None
        x_125 = torch.nn.functional.dropout(x_124, 0.0, False, False)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_127 = torch.nn.functional.dropout(x_126, 0.0, False, False)
        x_126 = None
        view_35 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_35 = x_127 * view_35
        x_127 = view_35 = None
        x_128 = x_122 + mul_35
        x_122 = mul_35 = None
        group_norm_36 = torch.nn.functional.group_norm(
            x_128,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        y_18 = torch._C._nn.avg_pool2d(group_norm_36, 3, 1, 1, False, False, None)
        sub_18 = y_18 - group_norm_36
        y_18 = group_norm_36 = None
        view_36 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_36 = sub_18 * view_36
        sub_18 = view_36 = None
        x_129 = x_128 + mul_36
        x_128 = mul_36 = None
        group_norm_37 = torch.nn.functional.group_norm(
            x_129,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_130 = torch.conv2d(
            group_norm_37,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_37 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_131 = torch._C._nn.gelu(x_130, approximate="none")
        x_130 = None
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_132 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_134 = torch.nn.functional.dropout(x_133, 0.0, False, False)
        x_133 = None
        view_37 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_37 = x_134 * view_37
        x_134 = view_37 = None
        x_135 = x_129 + mul_37
        x_129 = mul_37 = None
        group_norm_38 = torch.nn.functional.group_norm(
            x_135,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        y_19 = torch._C._nn.avg_pool2d(group_norm_38, 3, 1, 1, False, False, None)
        sub_19 = y_19 - group_norm_38
        y_19 = group_norm_38 = None
        view_38 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_38 = sub_19 * view_38
        sub_19 = view_38 = None
        x_136 = x_135 + mul_38
        x_135 = mul_38 = None
        group_norm_39 = torch.nn.functional.group_norm(
            x_136,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_137 = torch.conv2d(
            group_norm_39,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_39 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_138 = torch._C._nn.gelu(x_137, approximate="none")
        x_137 = None
        x_139 = torch.nn.functional.dropout(x_138, 0.0, False, False)
        x_138 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_141 = torch.nn.functional.dropout(x_140, 0.0, False, False)
        x_140 = None
        view_39 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_39 = x_141 * view_39
        x_141 = view_39 = None
        x_142 = x_136 + mul_39
        x_136 = mul_39 = None
        group_norm_40 = torch.nn.functional.group_norm(
            x_142,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        y_20 = torch._C._nn.avg_pool2d(group_norm_40, 3, 1, 1, False, False, None)
        sub_20 = y_20 - group_norm_40
        y_20 = group_norm_40 = None
        view_40 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_40 = sub_20 * view_40
        sub_20 = view_40 = None
        x_143 = x_142 + mul_40
        x_142 = mul_40 = None
        group_norm_41 = torch.nn.functional.group_norm(
            x_143,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_144 = torch.conv2d(
            group_norm_41,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_41 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_145 = torch._C._nn.gelu(x_144, approximate="none")
        x_144 = None
        x_146 = torch.nn.functional.dropout(x_145, 0.0, False, False)
        x_145 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_146 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_148 = torch.nn.functional.dropout(x_147, 0.0, False, False)
        x_147 = None
        view_41 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_41 = x_148 * view_41
        x_148 = view_41 = None
        x_149 = x_143 + mul_41
        x_143 = mul_41 = None
        group_norm_42 = torch.nn.functional.group_norm(
            x_149,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        y_21 = torch._C._nn.avg_pool2d(group_norm_42, 3, 1, 1, False, False, None)
        sub_21 = y_21 - group_norm_42
        y_21 = group_norm_42 = None
        view_42 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_42 = sub_21 * view_42
        sub_21 = view_42 = None
        x_150 = x_149 + mul_42
        x_149 = mul_42 = None
        group_norm_43 = torch.nn.functional.group_norm(
            x_150,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_151 = torch.conv2d(
            group_norm_43,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_43 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_152 = torch._C._nn.gelu(x_151, approximate="none")
        x_151 = None
        x_153 = torch.nn.functional.dropout(x_152, 0.0, False, False)
        x_152 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_155 = torch.nn.functional.dropout(x_154, 0.0, False, False)
        x_154 = None
        view_43 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_43 = x_155 * view_43
        x_155 = view_43 = None
        x_156 = x_150 + mul_43
        x_150 = mul_43 = None
        group_norm_44 = torch.nn.functional.group_norm(
            x_156,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (None)
        y_22 = torch._C._nn.avg_pool2d(group_norm_44, 3, 1, 1, False, False, None)
        sub_22 = y_22 - group_norm_44
        y_22 = group_norm_44 = None
        view_44 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_44 = sub_22 * view_44
        sub_22 = view_44 = None
        x_157 = x_156 + mul_44
        x_156 = mul_44 = None
        group_norm_45 = torch.nn.functional.group_norm(
            x_157,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (None)
        x_158 = torch.conv2d(
            group_norm_45,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_45 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_159 = torch._C._nn.gelu(x_158, approximate="none")
        x_158 = None
        x_160 = torch.nn.functional.dropout(x_159, 0.0, False, False)
        x_159 = None
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_160 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_162 = torch.nn.functional.dropout(x_161, 0.0, False, False)
        x_161 = None
        view_45 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_45 = x_162 * view_45
        x_162 = view_45 = None
        x_163 = x_157 + mul_45
        x_157 = mul_45 = None
        group_norm_46 = torch.nn.functional.group_norm(
            x_163,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (None)
        y_23 = torch._C._nn.avg_pool2d(group_norm_46, 3, 1, 1, False, False, None)
        sub_23 = y_23 - group_norm_46
        y_23 = group_norm_46 = None
        view_46 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_46 = sub_23 * view_46
        sub_23 = view_46 = None
        x_164 = x_163 + mul_46
        x_163 = mul_46 = None
        group_norm_47 = torch.nn.functional.group_norm(
            x_164,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (None)
        x_165 = torch.conv2d(
            group_norm_47,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_47 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_166 = torch._C._nn.gelu(x_165, approximate="none")
        x_165 = None
        x_167 = torch.nn.functional.dropout(x_166, 0.0, False, False)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_167 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_169 = torch.nn.functional.dropout(x_168, 0.0, False, False)
        x_168 = None
        view_47 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_47 = x_169 * view_47
        x_169 = view_47 = None
        x_170 = x_164 + mul_47
        x_164 = mul_47 = None
        group_norm_48 = torch.nn.functional.group_norm(
            x_170,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (None)
        y_24 = torch._C._nn.avg_pool2d(group_norm_48, 3, 1, 1, False, False, None)
        sub_24 = y_24 - group_norm_48
        y_24 = group_norm_48 = None
        view_48 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_48 = sub_24 * view_48
        sub_24 = view_48 = None
        x_171 = x_170 + mul_48
        x_170 = mul_48 = None
        group_norm_49 = torch.nn.functional.group_norm(
            x_171,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (None)
        x_172 = torch.conv2d(
            group_norm_49,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_49 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_173 = torch._C._nn.gelu(x_172, approximate="none")
        x_172 = None
        x_174 = torch.nn.functional.dropout(x_173, 0.0, False, False)
        x_173 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_176 = torch.nn.functional.dropout(x_175, 0.0, False, False)
        x_175 = None
        view_49 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_49 = x_176 * view_49
        x_176 = view_49 = None
        x_177 = x_171 + mul_49
        x_171 = mul_49 = None
        group_norm_50 = torch.nn.functional.group_norm(
            x_177,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (None)
        y_25 = torch._C._nn.avg_pool2d(group_norm_50, 3, 1, 1, False, False, None)
        sub_25 = y_25 - group_norm_50
        y_25 = group_norm_50 = None
        view_50 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_50 = sub_25 * view_50
        sub_25 = view_50 = None
        x_178 = x_177 + mul_50
        x_177 = mul_50 = None
        group_norm_51 = torch.nn.functional.group_norm(
            x_178,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (None)
        x_179 = torch.conv2d(
            group_norm_51,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_51 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_180 = torch._C._nn.gelu(x_179, approximate="none")
        x_179 = None
        x_181 = torch.nn.functional.dropout(x_180, 0.0, False, False)
        x_180 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_183 = torch.nn.functional.dropout(x_182, 0.0, False, False)
        x_182 = None
        view_51 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_51 = x_183 * view_51
        x_183 = view_51 = None
        x_184 = x_178 + mul_51
        x_178 = mul_51 = None
        group_norm_52 = torch.nn.functional.group_norm(
            x_184,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (None)
        y_26 = torch._C._nn.avg_pool2d(group_norm_52, 3, 1, 1, False, False, None)
        sub_26 = y_26 - group_norm_52
        y_26 = group_norm_52 = None
        view_52 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_52 = sub_26 * view_52
        sub_26 = view_52 = None
        x_185 = x_184 + mul_52
        x_184 = mul_52 = None
        group_norm_53 = torch.nn.functional.group_norm(
            x_185,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (None)
        x_186 = torch.conv2d(
            group_norm_53,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_53 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_187 = torch._C._nn.gelu(x_186, approximate="none")
        x_186 = None
        x_188 = torch.nn.functional.dropout(x_187, 0.0, False, False)
        x_187 = None
        x_189 = torch.conv2d(
            x_188,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_188 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_190 = torch.nn.functional.dropout(x_189, 0.0, False, False)
        x_189 = None
        view_53 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_53 = x_190 * view_53
        x_190 = view_53 = None
        x_191 = x_185 + mul_53
        x_185 = mul_53 = None
        group_norm_54 = torch.nn.functional.group_norm(
            x_191,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (None)
        y_27 = torch._C._nn.avg_pool2d(group_norm_54, 3, 1, 1, False, False, None)
        sub_27 = y_27 - group_norm_54
        y_27 = group_norm_54 = None
        view_54 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_54 = sub_27 * view_54
        sub_27 = view_54 = None
        x_192 = x_191 + mul_54
        x_191 = mul_54 = None
        group_norm_55 = torch.nn.functional.group_norm(
            x_192,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (None)
        x_193 = torch.conv2d(
            group_norm_55,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_55 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_194 = torch._C._nn.gelu(x_193, approximate="none")
        x_193 = None
        x_195 = torch.nn.functional.dropout(x_194, 0.0, False, False)
        x_194 = None
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_195 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_197 = torch.nn.functional.dropout(x_196, 0.0, False, False)
        x_196 = None
        view_55 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_55 = x_197 * view_55
        x_197 = view_55 = None
        x_198 = x_192 + mul_55
        x_192 = mul_55 = None
        group_norm_56 = torch.nn.functional.group_norm(
            x_198,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_ = (None)
        y_28 = torch._C._nn.avg_pool2d(group_norm_56, 3, 1, 1, False, False, None)
        sub_28 = y_28 - group_norm_56
        y_28 = group_norm_56 = None
        view_56 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_56 = sub_28 * view_56
        sub_28 = view_56 = None
        x_199 = x_198 + mul_56
        x_198 = mul_56 = None
        group_norm_57 = torch.nn.functional.group_norm(
            x_199,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_ = (None)
        x_200 = torch.conv2d(
            group_norm_57,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_57 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_201 = torch._C._nn.gelu(x_200, approximate="none")
        x_200 = None
        x_202 = torch.nn.functional.dropout(x_201, 0.0, False, False)
        x_201 = None
        x_203 = torch.conv2d(
            x_202,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_202 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_204 = torch.nn.functional.dropout(x_203, 0.0, False, False)
        x_203 = None
        view_57 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_57 = x_204 * view_57
        x_204 = view_57 = None
        x_205 = x_199 + mul_57
        x_199 = mul_57 = None
        group_norm_58 = torch.nn.functional.group_norm(
            x_205,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_ = (None)
        y_29 = torch._C._nn.avg_pool2d(group_norm_58, 3, 1, 1, False, False, None)
        sub_29 = y_29 - group_norm_58
        y_29 = group_norm_58 = None
        view_58 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_58 = sub_29 * view_58
        sub_29 = view_58 = None
        x_206 = x_205 + mul_58
        x_205 = mul_58 = None
        group_norm_59 = torch.nn.functional.group_norm(
            x_206,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_ = (None)
        x_207 = torch.conv2d(
            group_norm_59,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_59 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_208 = torch._C._nn.gelu(x_207, approximate="none")
        x_207 = None
        x_209 = torch.nn.functional.dropout(x_208, 0.0, False, False)
        x_208 = None
        x_210 = torch.conv2d(
            x_209,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_209 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_211 = torch.nn.functional.dropout(x_210, 0.0, False, False)
        x_210 = None
        view_59 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_59 = x_211 * view_59
        x_211 = view_59 = None
        x_212 = x_206 + mul_59
        x_206 = mul_59 = None
        group_norm_60 = torch.nn.functional.group_norm(
            x_212,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_ = (None)
        y_30 = torch._C._nn.avg_pool2d(group_norm_60, 3, 1, 1, False, False, None)
        sub_30 = y_30 - group_norm_60
        y_30 = group_norm_60 = None
        view_60 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_60 = sub_30 * view_60
        sub_30 = view_60 = None
        x_213 = x_212 + mul_60
        x_212 = mul_60 = None
        group_norm_61 = torch.nn.functional.group_norm(
            x_213,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_ = (None)
        x_214 = torch.conv2d(
            group_norm_61,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_61 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_215 = torch._C._nn.gelu(x_214, approximate="none")
        x_214 = None
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        x_217 = torch.conv2d(
            x_216,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_216 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_218 = torch.nn.functional.dropout(x_217, 0.0, False, False)
        x_217 = None
        view_61 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_61 = x_218 * view_61
        x_218 = view_61 = None
        x_219 = x_213 + mul_61
        x_213 = mul_61 = None
        group_norm_62 = torch.nn.functional.group_norm(
            x_219,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_ = (None)
        y_31 = torch._C._nn.avg_pool2d(group_norm_62, 3, 1, 1, False, False, None)
        sub_31 = y_31 - group_norm_62
        y_31 = group_norm_62 = None
        view_62 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_62 = sub_31 * view_62
        sub_31 = view_62 = None
        x_220 = x_219 + mul_62
        x_219 = mul_62 = None
        group_norm_63 = torch.nn.functional.group_norm(
            x_220,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_ = (None)
        x_221 = torch.conv2d(
            group_norm_63,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_63 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_222 = torch._C._nn.gelu(x_221, approximate="none")
        x_221 = None
        x_223 = torch.nn.functional.dropout(x_222, 0.0, False, False)
        x_222 = None
        x_224 = torch.conv2d(
            x_223,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_223 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_225 = torch.nn.functional.dropout(x_224, 0.0, False, False)
        x_224 = None
        view_63 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_63 = x_225 * view_63
        x_225 = view_63 = None
        x_226 = x_220 + mul_63
        x_220 = mul_63 = None
        group_norm_64 = torch.nn.functional.group_norm(
            x_226,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_ = (None)
        y_32 = torch._C._nn.avg_pool2d(group_norm_64, 3, 1, 1, False, False, None)
        sub_32 = y_32 - group_norm_64
        y_32 = group_norm_64 = None
        view_64 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_64 = sub_32 * view_64
        sub_32 = view_64 = None
        x_227 = x_226 + mul_64
        x_226 = mul_64 = None
        group_norm_65 = torch.nn.functional.group_norm(
            x_227,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_ = (None)
        x_228 = torch.conv2d(
            group_norm_65,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_65 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_229 = torch._C._nn.gelu(x_228, approximate="none")
        x_228 = None
        x_230 = torch.nn.functional.dropout(x_229, 0.0, False, False)
        x_229 = None
        x_231 = torch.conv2d(
            x_230,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_230 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_232 = torch.nn.functional.dropout(x_231, 0.0, False, False)
        x_231 = None
        view_65 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_65 = x_232 * view_65
        x_232 = view_65 = None
        x_233 = x_227 + mul_65
        x_227 = mul_65 = None
        group_norm_66 = torch.nn.functional.group_norm(
            x_233,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_ = (None)
        y_33 = torch._C._nn.avg_pool2d(group_norm_66, 3, 1, 1, False, False, None)
        sub_33 = y_33 - group_norm_66
        y_33 = group_norm_66 = None
        view_66 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_66 = sub_33 * view_66
        sub_33 = view_66 = None
        x_234 = x_233 + mul_66
        x_233 = mul_66 = None
        group_norm_67 = torch.nn.functional.group_norm(
            x_234,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_ = (None)
        x_235 = torch.conv2d(
            group_norm_67,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_67 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_236 = torch._C._nn.gelu(x_235, approximate="none")
        x_235 = None
        x_237 = torch.nn.functional.dropout(x_236, 0.0, False, False)
        x_236 = None
        x_238 = torch.conv2d(
            x_237,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_237 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_239 = torch.nn.functional.dropout(x_238, 0.0, False, False)
        x_238 = None
        view_67 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_67 = x_239 * view_67
        x_239 = view_67 = None
        x_240 = x_234 + mul_67
        x_234 = mul_67 = None
        group_norm_68 = torch.nn.functional.group_norm(
            x_240,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm1_parameters_bias_ = (None)
        y_34 = torch._C._nn.avg_pool2d(group_norm_68, 3, 1, 1, False, False, None)
        sub_34 = y_34 - group_norm_68
        y_34 = group_norm_68 = None
        view_68 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_68 = sub_34 * view_68
        sub_34 = view_68 = None
        x_241 = x_240 + mul_68
        x_240 = mul_68 = None
        group_norm_69 = torch.nn.functional.group_norm(
            x_241,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm2_parameters_bias_ = (None)
        x_242 = torch.conv2d(
            group_norm_69,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_69 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_243 = torch._C._nn.gelu(x_242, approximate="none")
        x_242 = None
        x_244 = torch.nn.functional.dropout(x_243, 0.0, False, False)
        x_243 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_244 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_246 = torch.nn.functional.dropout(x_245, 0.0, False, False)
        x_245 = None
        view_69 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_69 = x_246 * view_69
        x_246 = view_69 = None
        x_247 = x_241 + mul_69
        x_241 = mul_69 = None
        group_norm_70 = torch.nn.functional.group_norm(
            x_247,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm1_parameters_bias_ = (None)
        y_35 = torch._C._nn.avg_pool2d(group_norm_70, 3, 1, 1, False, False, None)
        sub_35 = y_35 - group_norm_70
        y_35 = group_norm_70 = None
        view_70 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_70 = sub_35 * view_70
        sub_35 = view_70 = None
        x_248 = x_247 + mul_70
        x_247 = mul_70 = None
        group_norm_71 = torch.nn.functional.group_norm(
            x_248,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm2_parameters_bias_ = (None)
        x_249 = torch.conv2d(
            group_norm_71,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_71 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_250 = torch._C._nn.gelu(x_249, approximate="none")
        x_249 = None
        x_251 = torch.nn.functional.dropout(x_250, 0.0, False, False)
        x_250 = None
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_251 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_253 = torch.nn.functional.dropout(x_252, 0.0, False, False)
        x_252 = None
        view_71 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_71 = x_253 * view_71
        x_253 = view_71 = None
        x_254 = x_248 + mul_71
        x_248 = mul_71 = None
        group_norm_72 = torch.nn.functional.group_norm(
            x_254,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm1_parameters_bias_ = (None)
        y_36 = torch._C._nn.avg_pool2d(group_norm_72, 3, 1, 1, False, False, None)
        sub_36 = y_36 - group_norm_72
        y_36 = group_norm_72 = None
        view_72 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_72 = sub_36 * view_72
        sub_36 = view_72 = None
        x_255 = x_254 + mul_72
        x_254 = mul_72 = None
        group_norm_73 = torch.nn.functional.group_norm(
            x_255,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm2_parameters_bias_ = (None)
        x_256 = torch.conv2d(
            group_norm_73,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_73 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_257 = torch._C._nn.gelu(x_256, approximate="none")
        x_256 = None
        x_258 = torch.nn.functional.dropout(x_257, 0.0, False, False)
        x_257 = None
        x_259 = torch.conv2d(
            x_258,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_258 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_260 = torch.nn.functional.dropout(x_259, 0.0, False, False)
        x_259 = None
        view_73 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_73 = x_260 * view_73
        x_260 = view_73 = None
        x_261 = x_255 + mul_73
        x_255 = mul_73 = None
        group_norm_74 = torch.nn.functional.group_norm(
            x_261,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm1_parameters_bias_ = (None)
        y_37 = torch._C._nn.avg_pool2d(group_norm_74, 3, 1, 1, False, False, None)
        sub_37 = y_37 - group_norm_74
        y_37 = group_norm_74 = None
        view_74 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_74 = sub_37 * view_74
        sub_37 = view_74 = None
        x_262 = x_261 + mul_74
        x_261 = mul_74 = None
        group_norm_75 = torch.nn.functional.group_norm(
            x_262,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm2_parameters_bias_ = (None)
        x_263 = torch.conv2d(
            group_norm_75,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_75 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_264 = torch._C._nn.gelu(x_263, approximate="none")
        x_263 = None
        x_265 = torch.nn.functional.dropout(x_264, 0.0, False, False)
        x_264 = None
        x_266 = torch.conv2d(
            x_265,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_265 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_267 = torch.nn.functional.dropout(x_266, 0.0, False, False)
        x_266 = None
        view_75 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_75 = x_267 * view_75
        x_267 = view_75 = None
        x_268 = x_262 + mul_75
        x_262 = mul_75 = None
        group_norm_76 = torch.nn.functional.group_norm(
            x_268,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm1_parameters_bias_ = (None)
        y_38 = torch._C._nn.avg_pool2d(group_norm_76, 3, 1, 1, False, False, None)
        sub_38 = y_38 - group_norm_76
        y_38 = group_norm_76 = None
        view_76 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_76 = sub_38 * view_76
        sub_38 = view_76 = None
        x_269 = x_268 + mul_76
        x_268 = mul_76 = None
        group_norm_77 = torch.nn.functional.group_norm(
            x_269,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm2_parameters_bias_ = (None)
        x_270 = torch.conv2d(
            group_norm_77,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_77 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_271 = torch._C._nn.gelu(x_270, approximate="none")
        x_270 = None
        x_272 = torch.nn.functional.dropout(x_271, 0.0, False, False)
        x_271 = None
        x_273 = torch.conv2d(
            x_272,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_272 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_274 = torch.nn.functional.dropout(x_273, 0.0, False, False)
        x_273 = None
        view_77 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_77 = x_274 * view_77
        x_274 = view_77 = None
        x_275 = x_269 + mul_77
        x_269 = mul_77 = None
        group_norm_78 = torch.nn.functional.group_norm(
            x_275,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm1_parameters_bias_ = (None)
        y_39 = torch._C._nn.avg_pool2d(group_norm_78, 3, 1, 1, False, False, None)
        sub_39 = y_39 - group_norm_78
        y_39 = group_norm_78 = None
        view_78 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_layer_scale1_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_78 = sub_39 * view_78
        sub_39 = view_78 = None
        x_276 = x_275 + mul_78
        x_275 = mul_78 = None
        group_norm_79 = torch.nn.functional.group_norm(
            x_276,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm2_parameters_bias_ = (None)
        x_277 = torch.conv2d(
            group_norm_79,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_79 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_278 = torch._C._nn.gelu(x_277, approximate="none")
        x_277 = None
        x_279 = torch.nn.functional.dropout(x_278, 0.0, False, False)
        x_278 = None
        x_280 = torch.conv2d(
            x_279,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_279 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_281 = torch.nn.functional.dropout(x_280, 0.0, False, False)
        x_280 = None
        view_79 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_layer_scale2_parameters_scale_.view(
            (384, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_79 = x_281 * view_79
        x_281 = view_79 = None
        x_282 = x_276 + mul_79
        x_276 = mul_79 = None
        x_283 = torch.conv2d(
            x_282,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_282 = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = (None)
        floordiv_3 = sym_sum // 32
        sym_sum = None
        sym_sum_4 = torch.sym_sum([1, floordiv_3])
        floordiv_3 = sym_sum_4 = None
        group_norm_80 = torch.nn.functional.group_norm(
            x_283,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        y_40 = torch._C._nn.avg_pool2d(group_norm_80, 3, 1, 1, False, False, None)
        sub_40 = y_40 - group_norm_80
        y_40 = group_norm_80 = None
        view_80 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_80 = sub_40 * view_80
        sub_40 = view_80 = None
        x_284 = x_283 + mul_80
        x_283 = mul_80 = None
        group_norm_81 = torch.nn.functional.group_norm(
            x_284,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_285 = torch.conv2d(
            group_norm_81,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_81 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_286 = torch._C._nn.gelu(x_285, approximate="none")
        x_285 = None
        x_287 = torch.nn.functional.dropout(x_286, 0.0, False, False)
        x_286 = None
        x_288 = torch.conv2d(
            x_287,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_287 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_289 = torch.nn.functional.dropout(x_288, 0.0, False, False)
        x_288 = None
        view_81 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_81 = x_289 * view_81
        x_289 = view_81 = None
        x_290 = x_284 + mul_81
        x_284 = mul_81 = None
        group_norm_82 = torch.nn.functional.group_norm(
            x_290,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        y_41 = torch._C._nn.avg_pool2d(group_norm_82, 3, 1, 1, False, False, None)
        sub_41 = y_41 - group_norm_82
        y_41 = group_norm_82 = None
        view_82 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_82 = sub_41 * view_82
        sub_41 = view_82 = None
        x_291 = x_290 + mul_82
        x_290 = mul_82 = None
        group_norm_83 = torch.nn.functional.group_norm(
            x_291,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_292 = torch.conv2d(
            group_norm_83,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_83 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_293 = torch._C._nn.gelu(x_292, approximate="none")
        x_292 = None
        x_294 = torch.nn.functional.dropout(x_293, 0.0, False, False)
        x_293 = None
        x_295 = torch.conv2d(
            x_294,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_294 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_296 = torch.nn.functional.dropout(x_295, 0.0, False, False)
        x_295 = None
        view_83 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_83 = x_296 * view_83
        x_296 = view_83 = None
        x_297 = x_291 + mul_83
        x_291 = mul_83 = None
        group_norm_84 = torch.nn.functional.group_norm(
            x_297,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        y_42 = torch._C._nn.avg_pool2d(group_norm_84, 3, 1, 1, False, False, None)
        sub_42 = y_42 - group_norm_84
        y_42 = group_norm_84 = None
        view_84 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_84 = sub_42 * view_84
        sub_42 = view_84 = None
        x_298 = x_297 + mul_84
        x_297 = mul_84 = None
        group_norm_85 = torch.nn.functional.group_norm(
            x_298,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_299 = torch.conv2d(
            group_norm_85,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_85 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_300 = torch._C._nn.gelu(x_299, approximate="none")
        x_299 = None
        x_301 = torch.nn.functional.dropout(x_300, 0.0, False, False)
        x_300 = None
        x_302 = torch.conv2d(
            x_301,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_301 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_303 = torch.nn.functional.dropout(x_302, 0.0, False, False)
        x_302 = None
        view_85 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_85 = x_303 * view_85
        x_303 = view_85 = None
        x_304 = x_298 + mul_85
        x_298 = mul_85 = None
        group_norm_86 = torch.nn.functional.group_norm(
            x_304,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        y_43 = torch._C._nn.avg_pool2d(group_norm_86, 3, 1, 1, False, False, None)
        sub_43 = y_43 - group_norm_86
        y_43 = group_norm_86 = None
        view_86 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_86 = sub_43 * view_86
        sub_43 = view_86 = None
        x_305 = x_304 + mul_86
        x_304 = mul_86 = None
        group_norm_87 = torch.nn.functional.group_norm(
            x_305,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_306 = torch.conv2d(
            group_norm_87,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_87 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_307 = torch._C._nn.gelu(x_306, approximate="none")
        x_306 = None
        x_308 = torch.nn.functional.dropout(x_307, 0.0, False, False)
        x_307 = None
        x_309 = torch.conv2d(
            x_308,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_308 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_310 = torch.nn.functional.dropout(x_309, 0.0, False, False)
        x_309 = None
        view_87 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_87 = x_310 * view_87
        x_310 = view_87 = None
        x_311 = x_305 + mul_87
        x_305 = mul_87 = None
        group_norm_88 = torch.nn.functional.group_norm(
            x_311,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        y_44 = torch._C._nn.avg_pool2d(group_norm_88, 3, 1, 1, False, False, None)
        sub_44 = y_44 - group_norm_88
        y_44 = group_norm_88 = None
        view_88 = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_88 = sub_44 * view_88
        sub_44 = view_88 = None
        x_312 = x_311 + mul_88
        x_311 = mul_88 = None
        group_norm_89 = torch.nn.functional.group_norm(
            x_312,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_313 = torch.conv2d(
            group_norm_89,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_89 = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_314 = torch._C._nn.gelu(x_313, approximate="none")
        x_313 = None
        x_315 = torch.nn.functional.dropout(x_314, 0.0, False, False)
        x_314 = None
        x_316 = torch.conv2d(
            x_315,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_315 = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_317 = torch.nn.functional.dropout(x_316, 0.0, False, False)
        x_316 = None
        view_89 = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_89 = x_317 * view_89
        x_317 = view_89 = None
        x_318 = x_312 + mul_89
        x_312 = mul_89 = None
        group_norm_90 = torch.nn.functional.group_norm(
            x_318,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        y_45 = torch._C._nn.avg_pool2d(group_norm_90, 3, 1, 1, False, False, None)
        sub_45 = y_45 - group_norm_90
        y_45 = group_norm_90 = None
        view_90 = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_90 = sub_45 * view_90
        sub_45 = view_90 = None
        x_319 = x_318 + mul_90
        x_318 = mul_90 = None
        group_norm_91 = torch.nn.functional.group_norm(
            x_319,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_320 = torch.conv2d(
            group_norm_91,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_91 = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_321 = torch._C._nn.gelu(x_320, approximate="none")
        x_320 = None
        x_322 = torch.nn.functional.dropout(x_321, 0.0, False, False)
        x_321 = None
        x_323 = torch.conv2d(
            x_322,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_322 = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_324 = torch.nn.functional.dropout(x_323, 0.0, False, False)
        x_323 = None
        view_91 = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_91 = x_324 * view_91
        x_324 = view_91 = None
        x_325 = x_319 + mul_91
        x_319 = mul_91 = None
        group_norm_92 = torch.nn.functional.group_norm(
            x_325,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (None)
        y_46 = torch._C._nn.avg_pool2d(group_norm_92, 3, 1, 1, False, False, None)
        sub_46 = y_46 - group_norm_92
        y_46 = group_norm_92 = None
        view_92 = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_92 = sub_46 * view_92
        sub_46 = view_92 = None
        x_326 = x_325 + mul_92
        x_325 = mul_92 = None
        group_norm_93 = torch.nn.functional.group_norm(
            x_326,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (None)
        x_327 = torch.conv2d(
            group_norm_93,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_93 = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_328 = torch._C._nn.gelu(x_327, approximate="none")
        x_327 = None
        x_329 = torch.nn.functional.dropout(x_328, 0.0, False, False)
        x_328 = None
        x_330 = torch.conv2d(
            x_329,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_329 = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_331 = torch.nn.functional.dropout(x_330, 0.0, False, False)
        x_330 = None
        view_93 = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_93 = x_331 * view_93
        x_331 = view_93 = None
        x_332 = x_326 + mul_93
        x_326 = mul_93 = None
        group_norm_94 = torch.nn.functional.group_norm(
            x_332,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (None)
        y_47 = torch._C._nn.avg_pool2d(group_norm_94, 3, 1, 1, False, False, None)
        sub_47 = y_47 - group_norm_94
        y_47 = group_norm_94 = None
        view_94 = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_layer_scale1_parameters_scale_ = (
            None
        )
        mul_94 = sub_47 * view_94
        sub_47 = view_94 = None
        x_333 = x_332 + mul_94
        x_332 = mul_94 = None
        group_norm_95 = torch.nn.functional.group_norm(
            x_333,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (None)
        x_334 = torch.conv2d(
            group_norm_95,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_95 = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_335 = torch._C._nn.gelu(x_334, approximate="none")
        x_334 = None
        x_336 = torch.nn.functional.dropout(x_335, 0.0, False, False)
        x_335 = None
        x_337 = torch.conv2d(
            x_336,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_336 = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_338 = torch.nn.functional.dropout(x_337, 0.0, False, False)
        x_337 = None
        view_95 = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_.view(
            (768, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_layer_scale2_parameters_scale_ = (
            None
        )
        mul_95 = x_338 * view_95
        x_338 = view_95 = None
        x_339 = x_333 + mul_95
        x_333 = mul_95 = None
        x_340 = torch.nn.functional.adaptive_avg_pool2d(x_339, 1)
        x_339 = None
        x_341 = x_340.permute(0, 2, 3, 1)
        x_340 = None
        x_342 = torch.nn.functional.layer_norm(
            x_341,
            (768,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_341 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_343 = x_342.permute(0, 3, 1, 2)
        x_342 = None
        x_344 = x_343.flatten(1, -1)
        x_343 = None
        x_345 = torch._C._nn.linear(
            x_344,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_344 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_345,)
