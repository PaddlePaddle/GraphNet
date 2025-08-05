import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_proj_parameters_weight_ = (
            L_self_modules_stem_modules_proj_parameters_weight_
        )
        l_self_modules_stem_modules_proj_parameters_bias_ = (
            L_self_modules_stem_modules_proj_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_norm_parameters_weight_ = (
            L_self_modules_stem_modules_norm_parameters_weight_
        )
        l_self_modules_stem_modules_norm_parameters_bias_ = (
            L_self_modules_stem_modules_norm_parameters_bias_
        )
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_norm_parameters_bias_ = L_self_modules_norm_parameters_bias_
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_proj_parameters_weight_,
            l_self_modules_stem_modules_proj_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_stem_modules_proj_parameters_weight_
        ) = l_self_modules_stem_modules_proj_parameters_bias_ = None
        x_1 = x.permute(0, 2, 3, 1)
        x = None
        x_2 = torch.nn.functional.layer_norm(
            x_1,
            (96,),
            l_self_modules_stem_modules_norm_parameters_weight_,
            l_self_modules_stem_modules_norm_parameters_bias_,
            1e-05,
        )
        x_1 = (
            l_self_modules_stem_modules_norm_parameters_weight_
        ) = l_self_modules_stem_modules_norm_parameters_bias_ = None
        x_3 = x_2.permute(0, 3, 1, 2)
        x_2 = None
        x_4 = x_3.permute(0, 2, 3, 1)
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (96,),
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_4 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_6 = x_5.permute(0, 3, 1, 2)
        x_5 = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = (None)
        split = torch.functional.split(x_7, [96, 96, 3], 1)
        x_7 = None
        q = split[0]
        ctx = split[1]
        gates = split[2]
        split = None
        input_1 = torch.conv2d(
            ctx,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        ctx = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_2 = torch._C._nn.gelu(input_1, approximate="none")
        input_1 = None
        getitem_3 = gates[(slice(None, None, None), slice(0, 1, None))]
        mul = input_2 * getitem_3
        getitem_3 = None
        ctx_all = 0 + mul
        mul = None
        input_3 = torch.conv2d(
            input_2,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        input_2 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_4 = torch._C._nn.gelu(input_3, approximate="none")
        input_3 = None
        getitem_4 = gates[(slice(None, None, None), slice(1, 2, None))]
        mul_1 = input_4 * getitem_4
        getitem_4 = None
        ctx_all_1 = ctx_all + mul_1
        ctx_all = mul_1 = None
        mean = input_4.mean((2, 3), keepdim=True)
        input_4 = None
        ctx_global = torch._C._nn.gelu(mean, approximate="none")
        mean = None
        getitem_5 = gates[(slice(None, None, None), slice(2, None, None))]
        gates = None
        mul_2 = ctx_global * getitem_5
        ctx_global = getitem_5 = None
        ctx_all_2 = ctx_all_1 + mul_2
        ctx_all_1 = mul_2 = None
        conv2d_4 = torch.conv2d(
            ctx_all_2,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_2 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out = q * conv2d_4
        q = conv2d_4 = None
        x_out_1 = torch.conv2d(
            x_out,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_2 = torch.nn.functional.dropout(x_out_1, 0.0, False, False)
        x_out_1 = None
        x_8 = x_3 + x_out_2
        x_3 = x_out_2 = None
        x_9 = x_8.permute(0, 2, 3, 1)
        x_10 = torch.nn.functional.layer_norm(
            x_9,
            (96,),
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_9 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_11 = x_10.permute(0, 3, 1, 2)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_13 = torch._C._nn.gelu(x_12, approximate="none")
        x_12 = None
        x_14 = torch.nn.functional.dropout(x_13, 0.0, False, False)
        x_13 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_16 = torch.nn.functional.dropout(x_15, 0.0, False, False)
        x_15 = None
        x_17 = x_8 + x_16
        x_8 = x_16 = None
        x_18 = x_17.permute(0, 2, 3, 1)
        x_19 = torch.nn.functional.layer_norm(
            x_18,
            (96,),
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_18 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_20 = x_19.permute(0, 3, 1, 2)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = (None)
        split_1 = torch.functional.split(x_21, [96, 96, 3], 1)
        x_21 = None
        q_1 = split_1[0]
        ctx_1 = split_1[1]
        gates_1 = split_1[2]
        split_1 = None
        input_5 = torch.conv2d(
            ctx_1,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        ctx_1 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_6 = torch._C._nn.gelu(input_5, approximate="none")
        input_5 = None
        getitem_9 = gates_1[(slice(None, None, None), slice(0, 1, None))]
        mul_4 = input_6 * getitem_9
        getitem_9 = None
        ctx_all_3 = 0 + mul_4
        mul_4 = None
        input_7 = torch.conv2d(
            input_6,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        input_6 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_8 = torch._C._nn.gelu(input_7, approximate="none")
        input_7 = None
        getitem_10 = gates_1[(slice(None, None, None), slice(1, 2, None))]
        mul_5 = input_8 * getitem_10
        getitem_10 = None
        ctx_all_4 = ctx_all_3 + mul_5
        ctx_all_3 = mul_5 = None
        mean_1 = input_8.mean((2, 3), keepdim=True)
        input_8 = None
        ctx_global_1 = torch._C._nn.gelu(mean_1, approximate="none")
        mean_1 = None
        getitem_11 = gates_1[(slice(None, None, None), slice(2, None, None))]
        gates_1 = None
        mul_6 = ctx_global_1 * getitem_11
        ctx_global_1 = getitem_11 = None
        ctx_all_5 = ctx_all_4 + mul_6
        ctx_all_4 = mul_6 = None
        conv2d_11 = torch.conv2d(
            ctx_all_5,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_5 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_3 = q_1 * conv2d_11
        q_1 = conv2d_11 = None
        x_out_4 = torch.conv2d(
            x_out_3,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_3 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_5 = torch.nn.functional.dropout(x_out_4, 0.0, False, False)
        x_out_4 = None
        x_22 = x_17 + x_out_5
        x_17 = x_out_5 = None
        x_23 = x_22.permute(0, 2, 3, 1)
        x_24 = torch.nn.functional.layer_norm(
            x_23,
            (96,),
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_23 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_25 = x_24.permute(0, 3, 1, 2)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_27 = torch._C._nn.gelu(x_26, approximate="none")
        x_26 = None
        x_28 = torch.nn.functional.dropout(x_27, 0.0, False, False)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_30 = torch.nn.functional.dropout(x_29, 0.0, False, False)
        x_29 = None
        x_31 = x_22 + x_30
        x_22 = x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_bias_ = (None)
        x_33 = x_32.permute(0, 2, 3, 1)
        x_32 = None
        x_34 = torch.nn.functional.layer_norm(
            x_33,
            (192,),
            l_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_33 = l_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_35 = x_34.permute(0, 3, 1, 2)
        x_34 = None
        x_36 = x_35.permute(0, 2, 3, 1)
        x_37 = torch.nn.functional.layer_norm(
            x_36,
            (192,),
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_36 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_38 = x_37.permute(0, 3, 1, 2)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = (None)
        split_2 = torch.functional.split(x_39, [192, 192, 3], 1)
        x_39 = None
        q_2 = split_2[0]
        ctx_2 = split_2[1]
        gates_2 = split_2[2]
        split_2 = None
        input_9 = torch.conv2d(
            ctx_2,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        ctx_2 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_10 = torch._C._nn.gelu(input_9, approximate="none")
        input_9 = None
        getitem_15 = gates_2[(slice(None, None, None), slice(0, 1, None))]
        mul_8 = input_10 * getitem_15
        getitem_15 = None
        ctx_all_6 = 0 + mul_8
        mul_8 = None
        input_11 = torch.conv2d(
            input_10,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        input_10 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_12 = torch._C._nn.gelu(input_11, approximate="none")
        input_11 = None
        getitem_16 = gates_2[(slice(None, None, None), slice(1, 2, None))]
        mul_9 = input_12 * getitem_16
        getitem_16 = None
        ctx_all_7 = ctx_all_6 + mul_9
        ctx_all_6 = mul_9 = None
        mean_2 = input_12.mean((2, 3), keepdim=True)
        input_12 = None
        ctx_global_2 = torch._C._nn.gelu(mean_2, approximate="none")
        mean_2 = None
        getitem_17 = gates_2[(slice(None, None, None), slice(2, None, None))]
        gates_2 = None
        mul_10 = ctx_global_2 * getitem_17
        ctx_global_2 = getitem_17 = None
        ctx_all_8 = ctx_all_7 + mul_10
        ctx_all_7 = mul_10 = None
        conv2d_19 = torch.conv2d(
            ctx_all_8,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_8 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_6 = q_2 * conv2d_19
        q_2 = conv2d_19 = None
        x_out_7 = torch.conv2d(
            x_out_6,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_6 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_8 = torch.nn.functional.dropout(x_out_7, 0.0, False, False)
        x_out_7 = None
        x_40 = x_35 + x_out_8
        x_35 = x_out_8 = None
        x_41 = x_40.permute(0, 2, 3, 1)
        x_42 = torch.nn.functional.layer_norm(
            x_41,
            (192,),
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_41 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_43 = x_42.permute(0, 3, 1, 2)
        x_42 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_45 = torch._C._nn.gelu(x_44, approximate="none")
        x_44 = None
        x_46 = torch.nn.functional.dropout(x_45, 0.0, False, False)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_48 = torch.nn.functional.dropout(x_47, 0.0, False, False)
        x_47 = None
        x_49 = x_40 + x_48
        x_40 = x_48 = None
        x_50 = x_49.permute(0, 2, 3, 1)
        x_51 = torch.nn.functional.layer_norm(
            x_50,
            (192,),
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_50 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_52 = x_51.permute(0, 3, 1, 2)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = (None)
        split_3 = torch.functional.split(x_53, [192, 192, 3], 1)
        x_53 = None
        q_3 = split_3[0]
        ctx_3 = split_3[1]
        gates_3 = split_3[2]
        split_3 = None
        input_13 = torch.conv2d(
            ctx_3,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        ctx_3 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_14 = torch._C._nn.gelu(input_13, approximate="none")
        input_13 = None
        getitem_21 = gates_3[(slice(None, None, None), slice(0, 1, None))]
        mul_12 = input_14 * getitem_21
        getitem_21 = None
        ctx_all_9 = 0 + mul_12
        mul_12 = None
        input_15 = torch.conv2d(
            input_14,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        input_14 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_16 = torch._C._nn.gelu(input_15, approximate="none")
        input_15 = None
        getitem_22 = gates_3[(slice(None, None, None), slice(1, 2, None))]
        mul_13 = input_16 * getitem_22
        getitem_22 = None
        ctx_all_10 = ctx_all_9 + mul_13
        ctx_all_9 = mul_13 = None
        mean_3 = input_16.mean((2, 3), keepdim=True)
        input_16 = None
        ctx_global_3 = torch._C._nn.gelu(mean_3, approximate="none")
        mean_3 = None
        getitem_23 = gates_3[(slice(None, None, None), slice(2, None, None))]
        gates_3 = None
        mul_14 = ctx_global_3 * getitem_23
        ctx_global_3 = getitem_23 = None
        ctx_all_11 = ctx_all_10 + mul_14
        ctx_all_10 = mul_14 = None
        conv2d_26 = torch.conv2d(
            ctx_all_11,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_11 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_9 = q_3 * conv2d_26
        q_3 = conv2d_26 = None
        x_out_10 = torch.conv2d(
            x_out_9,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_9 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_11 = torch.nn.functional.dropout(x_out_10, 0.0, False, False)
        x_out_10 = None
        x_54 = x_49 + x_out_11
        x_49 = x_out_11 = None
        x_55 = x_54.permute(0, 2, 3, 1)
        x_56 = torch.nn.functional.layer_norm(
            x_55,
            (192,),
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_55 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_57 = x_56.permute(0, 3, 1, 2)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_59 = torch._C._nn.gelu(x_58, approximate="none")
        x_58 = None
        x_60 = torch.nn.functional.dropout(x_59, 0.0, False, False)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_62 = torch.nn.functional.dropout(x_61, 0.0, False, False)
        x_61 = None
        x_63 = x_54 + x_62
        x_54 = x_62 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_bias_ = (None)
        x_65 = x_64.permute(0, 2, 3, 1)
        x_64 = None
        x_66 = torch.nn.functional.layer_norm(
            x_65,
            (384,),
            l_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_65 = l_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_67 = x_66.permute(0, 3, 1, 2)
        x_66 = None
        x_68 = x_67.permute(0, 2, 3, 1)
        x_69 = torch.nn.functional.layer_norm(
            x_68,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_68 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_70 = x_69.permute(0, 3, 1, 2)
        x_69 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = (None)
        split_4 = torch.functional.split(x_71, [384, 384, 3], 1)
        x_71 = None
        q_4 = split_4[0]
        ctx_4 = split_4[1]
        gates_4 = split_4[2]
        split_4 = None
        input_17 = torch.conv2d(
            ctx_4,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_4 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_18 = torch._C._nn.gelu(input_17, approximate="none")
        input_17 = None
        getitem_27 = gates_4[(slice(None, None, None), slice(0, 1, None))]
        mul_16 = input_18 * getitem_27
        getitem_27 = None
        ctx_all_12 = 0 + mul_16
        mul_16 = None
        input_19 = torch.conv2d(
            input_18,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_18 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_20 = torch._C._nn.gelu(input_19, approximate="none")
        input_19 = None
        getitem_28 = gates_4[(slice(None, None, None), slice(1, 2, None))]
        mul_17 = input_20 * getitem_28
        getitem_28 = None
        ctx_all_13 = ctx_all_12 + mul_17
        ctx_all_12 = mul_17 = None
        mean_4 = input_20.mean((2, 3), keepdim=True)
        input_20 = None
        ctx_global_4 = torch._C._nn.gelu(mean_4, approximate="none")
        mean_4 = None
        getitem_29 = gates_4[(slice(None, None, None), slice(2, None, None))]
        gates_4 = None
        mul_18 = ctx_global_4 * getitem_29
        ctx_global_4 = getitem_29 = None
        ctx_all_14 = ctx_all_13 + mul_18
        ctx_all_13 = mul_18 = None
        conv2d_34 = torch.conv2d(
            ctx_all_14,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_14 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_12 = q_4 * conv2d_34
        q_4 = conv2d_34 = None
        x_out_13 = torch.conv2d(
            x_out_12,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_12 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_14 = torch.nn.functional.dropout(x_out_13, 0.0, False, False)
        x_out_13 = None
        x_72 = x_67 + x_out_14
        x_67 = x_out_14 = None
        x_73 = x_72.permute(0, 2, 3, 1)
        x_74 = torch.nn.functional.layer_norm(
            x_73,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_73 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_75 = x_74.permute(0, 3, 1, 2)
        x_74 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_77 = torch._C._nn.gelu(x_76, approximate="none")
        x_76 = None
        x_78 = torch.nn.functional.dropout(x_77, 0.0, False, False)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_80 = torch.nn.functional.dropout(x_79, 0.0, False, False)
        x_79 = None
        x_81 = x_72 + x_80
        x_72 = x_80 = None
        x_82 = x_81.permute(0, 2, 3, 1)
        x_83 = torch.nn.functional.layer_norm(
            x_82,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_82 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_84 = x_83.permute(0, 3, 1, 2)
        x_83 = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_84 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = (None)
        split_5 = torch.functional.split(x_85, [384, 384, 3], 1)
        x_85 = None
        q_5 = split_5[0]
        ctx_5 = split_5[1]
        gates_5 = split_5[2]
        split_5 = None
        input_21 = torch.conv2d(
            ctx_5,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_5 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_22 = torch._C._nn.gelu(input_21, approximate="none")
        input_21 = None
        getitem_33 = gates_5[(slice(None, None, None), slice(0, 1, None))]
        mul_20 = input_22 * getitem_33
        getitem_33 = None
        ctx_all_15 = 0 + mul_20
        mul_20 = None
        input_23 = torch.conv2d(
            input_22,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_22 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_24 = torch._C._nn.gelu(input_23, approximate="none")
        input_23 = None
        getitem_34 = gates_5[(slice(None, None, None), slice(1, 2, None))]
        mul_21 = input_24 * getitem_34
        getitem_34 = None
        ctx_all_16 = ctx_all_15 + mul_21
        ctx_all_15 = mul_21 = None
        mean_5 = input_24.mean((2, 3), keepdim=True)
        input_24 = None
        ctx_global_5 = torch._C._nn.gelu(mean_5, approximate="none")
        mean_5 = None
        getitem_35 = gates_5[(slice(None, None, None), slice(2, None, None))]
        gates_5 = None
        mul_22 = ctx_global_5 * getitem_35
        ctx_global_5 = getitem_35 = None
        ctx_all_17 = ctx_all_16 + mul_22
        ctx_all_16 = mul_22 = None
        conv2d_41 = torch.conv2d(
            ctx_all_17,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_17 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_15 = q_5 * conv2d_41
        q_5 = conv2d_41 = None
        x_out_16 = torch.conv2d(
            x_out_15,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_15 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_17 = torch.nn.functional.dropout(x_out_16, 0.0, False, False)
        x_out_16 = None
        x_86 = x_81 + x_out_17
        x_81 = x_out_17 = None
        x_87 = x_86.permute(0, 2, 3, 1)
        x_88 = torch.nn.functional.layer_norm(
            x_87,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_87 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_89 = x_88.permute(0, 3, 1, 2)
        x_88 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_91 = torch._C._nn.gelu(x_90, approximate="none")
        x_90 = None
        x_92 = torch.nn.functional.dropout(x_91, 0.0, False, False)
        x_91 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_92 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_94 = torch.nn.functional.dropout(x_93, 0.0, False, False)
        x_93 = None
        x_95 = x_86 + x_94
        x_86 = x_94 = None
        x_96 = x_95.permute(0, 2, 3, 1)
        x_97 = torch.nn.functional.layer_norm(
            x_96,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_96 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_98 = x_97.permute(0, 3, 1, 2)
        x_97 = None
        x_99 = torch.conv2d(
            x_98,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_98 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_bias_ = (None)
        split_6 = torch.functional.split(x_99, [384, 384, 3], 1)
        x_99 = None
        q_6 = split_6[0]
        ctx_6 = split_6[1]
        gates_6 = split_6[2]
        split_6 = None
        input_25 = torch.conv2d(
            ctx_6,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_6 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_26 = torch._C._nn.gelu(input_25, approximate="none")
        input_25 = None
        getitem_39 = gates_6[(slice(None, None, None), slice(0, 1, None))]
        mul_24 = input_26 * getitem_39
        getitem_39 = None
        ctx_all_18 = 0 + mul_24
        mul_24 = None
        input_27 = torch.conv2d(
            input_26,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_26 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_28 = torch._C._nn.gelu(input_27, approximate="none")
        input_27 = None
        getitem_40 = gates_6[(slice(None, None, None), slice(1, 2, None))]
        mul_25 = input_28 * getitem_40
        getitem_40 = None
        ctx_all_19 = ctx_all_18 + mul_25
        ctx_all_18 = mul_25 = None
        mean_6 = input_28.mean((2, 3), keepdim=True)
        input_28 = None
        ctx_global_6 = torch._C._nn.gelu(mean_6, approximate="none")
        mean_6 = None
        getitem_41 = gates_6[(slice(None, None, None), slice(2, None, None))]
        gates_6 = None
        mul_26 = ctx_global_6 * getitem_41
        ctx_global_6 = getitem_41 = None
        ctx_all_20 = ctx_all_19 + mul_26
        ctx_all_19 = mul_26 = None
        conv2d_48 = torch.conv2d(
            ctx_all_20,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_20 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_18 = q_6 * conv2d_48
        q_6 = conv2d_48 = None
        x_out_19 = torch.conv2d(
            x_out_18,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_18 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_20 = torch.nn.functional.dropout(x_out_19, 0.0, False, False)
        x_out_19 = None
        x_100 = x_95 + x_out_20
        x_95 = x_out_20 = None
        x_101 = x_100.permute(0, 2, 3, 1)
        x_102 = torch.nn.functional.layer_norm(
            x_101,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_101 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_103 = x_102.permute(0, 3, 1, 2)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_105 = torch._C._nn.gelu(x_104, approximate="none")
        x_104 = None
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_108 = torch.nn.functional.dropout(x_107, 0.0, False, False)
        x_107 = None
        x_109 = x_100 + x_108
        x_100 = x_108 = None
        x_110 = x_109.permute(0, 2, 3, 1)
        x_111 = torch.nn.functional.layer_norm(
            x_110,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_110 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_112 = x_111.permute(0, 3, 1, 2)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_bias_ = (None)
        split_7 = torch.functional.split(x_113, [384, 384, 3], 1)
        x_113 = None
        q_7 = split_7[0]
        ctx_7 = split_7[1]
        gates_7 = split_7[2]
        split_7 = None
        input_29 = torch.conv2d(
            ctx_7,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_7 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_30 = torch._C._nn.gelu(input_29, approximate="none")
        input_29 = None
        getitem_45 = gates_7[(slice(None, None, None), slice(0, 1, None))]
        mul_28 = input_30 * getitem_45
        getitem_45 = None
        ctx_all_21 = 0 + mul_28
        mul_28 = None
        input_31 = torch.conv2d(
            input_30,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_30 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_32 = torch._C._nn.gelu(input_31, approximate="none")
        input_31 = None
        getitem_46 = gates_7[(slice(None, None, None), slice(1, 2, None))]
        mul_29 = input_32 * getitem_46
        getitem_46 = None
        ctx_all_22 = ctx_all_21 + mul_29
        ctx_all_21 = mul_29 = None
        mean_7 = input_32.mean((2, 3), keepdim=True)
        input_32 = None
        ctx_global_7 = torch._C._nn.gelu(mean_7, approximate="none")
        mean_7 = None
        getitem_47 = gates_7[(slice(None, None, None), slice(2, None, None))]
        gates_7 = None
        mul_30 = ctx_global_7 * getitem_47
        ctx_global_7 = getitem_47 = None
        ctx_all_23 = ctx_all_22 + mul_30
        ctx_all_22 = mul_30 = None
        conv2d_55 = torch.conv2d(
            ctx_all_23,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_23 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_21 = q_7 * conv2d_55
        q_7 = conv2d_55 = None
        x_out_22 = torch.conv2d(
            x_out_21,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_21 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_23 = torch.nn.functional.dropout(x_out_22, 0.0, False, False)
        x_out_22 = None
        x_114 = x_109 + x_out_23
        x_109 = x_out_23 = None
        x_115 = x_114.permute(0, 2, 3, 1)
        x_116 = torch.nn.functional.layer_norm(
            x_115,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_115 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_117 = x_116.permute(0, 3, 1, 2)
        x_116 = None
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_119 = torch._C._nn.gelu(x_118, approximate="none")
        x_118 = None
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_120 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_122 = torch.nn.functional.dropout(x_121, 0.0, False, False)
        x_121 = None
        x_123 = x_114 + x_122
        x_114 = x_122 = None
        x_124 = x_123.permute(0, 2, 3, 1)
        x_125 = torch.nn.functional.layer_norm(
            x_124,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_124 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        x_126 = x_125.permute(0, 3, 1, 2)
        x_125 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_126 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_bias_ = (None)
        split_8 = torch.functional.split(x_127, [384, 384, 3], 1)
        x_127 = None
        q_8 = split_8[0]
        ctx_8 = split_8[1]
        gates_8 = split_8[2]
        split_8 = None
        input_33 = torch.conv2d(
            ctx_8,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_8 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_34 = torch._C._nn.gelu(input_33, approximate="none")
        input_33 = None
        getitem_51 = gates_8[(slice(None, None, None), slice(0, 1, None))]
        mul_32 = input_34 * getitem_51
        getitem_51 = None
        ctx_all_24 = 0 + mul_32
        mul_32 = None
        input_35 = torch.conv2d(
            input_34,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_34 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_36 = torch._C._nn.gelu(input_35, approximate="none")
        input_35 = None
        getitem_52 = gates_8[(slice(None, None, None), slice(1, 2, None))]
        mul_33 = input_36 * getitem_52
        getitem_52 = None
        ctx_all_25 = ctx_all_24 + mul_33
        ctx_all_24 = mul_33 = None
        mean_8 = input_36.mean((2, 3), keepdim=True)
        input_36 = None
        ctx_global_8 = torch._C._nn.gelu(mean_8, approximate="none")
        mean_8 = None
        getitem_53 = gates_8[(slice(None, None, None), slice(2, None, None))]
        gates_8 = None
        mul_34 = ctx_global_8 * getitem_53
        ctx_global_8 = getitem_53 = None
        ctx_all_26 = ctx_all_25 + mul_34
        ctx_all_25 = mul_34 = None
        conv2d_62 = torch.conv2d(
            ctx_all_26,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_26 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_24 = q_8 * conv2d_62
        q_8 = conv2d_62 = None
        x_out_25 = torch.conv2d(
            x_out_24,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_24 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_26 = torch.nn.functional.dropout(x_out_25, 0.0, False, False)
        x_out_25 = None
        x_128 = x_123 + x_out_26
        x_123 = x_out_26 = None
        x_129 = x_128.permute(0, 2, 3, 1)
        x_130 = torch.nn.functional.layer_norm(
            x_129,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_129 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_131 = x_130.permute(0, 3, 1, 2)
        x_130 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_131 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_133 = torch._C._nn.gelu(x_132, approximate="none")
        x_132 = None
        x_134 = torch.nn.functional.dropout(x_133, 0.0, False, False)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_134 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_136 = torch.nn.functional.dropout(x_135, 0.0, False, False)
        x_135 = None
        x_137 = x_128 + x_136
        x_128 = x_136 = None
        x_138 = x_137.permute(0, 2, 3, 1)
        x_139 = torch.nn.functional.layer_norm(
            x_138,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_138 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        x_140 = x_139.permute(0, 3, 1, 2)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_140 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_bias_ = (None)
        split_9 = torch.functional.split(x_141, [384, 384, 3], 1)
        x_141 = None
        q_9 = split_9[0]
        ctx_9 = split_9[1]
        gates_9 = split_9[2]
        split_9 = None
        input_37 = torch.conv2d(
            ctx_9,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_9 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_38 = torch._C._nn.gelu(input_37, approximate="none")
        input_37 = None
        getitem_57 = gates_9[(slice(None, None, None), slice(0, 1, None))]
        mul_36 = input_38 * getitem_57
        getitem_57 = None
        ctx_all_27 = 0 + mul_36
        mul_36 = None
        input_39 = torch.conv2d(
            input_38,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_38 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_40 = torch._C._nn.gelu(input_39, approximate="none")
        input_39 = None
        getitem_58 = gates_9[(slice(None, None, None), slice(1, 2, None))]
        mul_37 = input_40 * getitem_58
        getitem_58 = None
        ctx_all_28 = ctx_all_27 + mul_37
        ctx_all_27 = mul_37 = None
        mean_9 = input_40.mean((2, 3), keepdim=True)
        input_40 = None
        ctx_global_9 = torch._C._nn.gelu(mean_9, approximate="none")
        mean_9 = None
        getitem_59 = gates_9[(slice(None, None, None), slice(2, None, None))]
        gates_9 = None
        mul_38 = ctx_global_9 * getitem_59
        ctx_global_9 = getitem_59 = None
        ctx_all_29 = ctx_all_28 + mul_38
        ctx_all_28 = mul_38 = None
        conv2d_69 = torch.conv2d(
            ctx_all_29,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_29 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_27 = q_9 * conv2d_69
        q_9 = conv2d_69 = None
        x_out_28 = torch.conv2d(
            x_out_27,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_27 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_29 = torch.nn.functional.dropout(x_out_28, 0.0, False, False)
        x_out_28 = None
        x_142 = x_137 + x_out_29
        x_137 = x_out_29 = None
        x_143 = x_142.permute(0, 2, 3, 1)
        x_144 = torch.nn.functional.layer_norm(
            x_143,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_143 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_145 = x_144.permute(0, 3, 1, 2)
        x_144 = None
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_145 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_147 = torch._C._nn.gelu(x_146, approximate="none")
        x_146 = None
        x_148 = torch.nn.functional.dropout(x_147, 0.0, False, False)
        x_147 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_150 = torch.nn.functional.dropout(x_149, 0.0, False, False)
        x_149 = None
        x_151 = x_142 + x_150
        x_142 = x_150 = None
        x_152 = x_151.permute(0, 2, 3, 1)
        x_153 = torch.nn.functional.layer_norm(
            x_152,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_152 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (None)
        x_154 = x_153.permute(0, 3, 1, 2)
        x_153 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_bias_ = (None)
        split_10 = torch.functional.split(x_155, [384, 384, 3], 1)
        x_155 = None
        q_10 = split_10[0]
        ctx_10 = split_10[1]
        gates_10 = split_10[2]
        split_10 = None
        input_41 = torch.conv2d(
            ctx_10,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_10 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_42 = torch._C._nn.gelu(input_41, approximate="none")
        input_41 = None
        getitem_63 = gates_10[(slice(None, None, None), slice(0, 1, None))]
        mul_40 = input_42 * getitem_63
        getitem_63 = None
        ctx_all_30 = 0 + mul_40
        mul_40 = None
        input_43 = torch.conv2d(
            input_42,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_42 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_44 = torch._C._nn.gelu(input_43, approximate="none")
        input_43 = None
        getitem_64 = gates_10[(slice(None, None, None), slice(1, 2, None))]
        mul_41 = input_44 * getitem_64
        getitem_64 = None
        ctx_all_31 = ctx_all_30 + mul_41
        ctx_all_30 = mul_41 = None
        mean_10 = input_44.mean((2, 3), keepdim=True)
        input_44 = None
        ctx_global_10 = torch._C._nn.gelu(mean_10, approximate="none")
        mean_10 = None
        getitem_65 = gates_10[(slice(None, None, None), slice(2, None, None))]
        gates_10 = None
        mul_42 = ctx_global_10 * getitem_65
        ctx_global_10 = getitem_65 = None
        ctx_all_32 = ctx_all_31 + mul_42
        ctx_all_31 = mul_42 = None
        conv2d_76 = torch.conv2d(
            ctx_all_32,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_32 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_30 = q_10 * conv2d_76
        q_10 = conv2d_76 = None
        x_out_31 = torch.conv2d(
            x_out_30,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_30 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_32 = torch.nn.functional.dropout(x_out_31, 0.0, False, False)
        x_out_31 = None
        x_156 = x_151 + x_out_32
        x_151 = x_out_32 = None
        x_157 = x_156.permute(0, 2, 3, 1)
        x_158 = torch.nn.functional.layer_norm(
            x_157,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_157 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (None)
        x_159 = x_158.permute(0, 3, 1, 2)
        x_158 = None
        x_160 = torch.conv2d(
            x_159,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_159 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_161 = torch._C._nn.gelu(x_160, approximate="none")
        x_160 = None
        x_162 = torch.nn.functional.dropout(x_161, 0.0, False, False)
        x_161 = None
        x_163 = torch.conv2d(
            x_162,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_162 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_164 = torch.nn.functional.dropout(x_163, 0.0, False, False)
        x_163 = None
        x_165 = x_156 + x_164
        x_156 = x_164 = None
        x_166 = x_165.permute(0, 2, 3, 1)
        x_167 = torch.nn.functional.layer_norm(
            x_166,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_166 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (None)
        x_168 = x_167.permute(0, 3, 1, 2)
        x_167 = None
        x_169 = torch.conv2d(
            x_168,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_168 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_bias_ = (None)
        split_11 = torch.functional.split(x_169, [384, 384, 3], 1)
        x_169 = None
        q_11 = split_11[0]
        ctx_11 = split_11[1]
        gates_11 = split_11[2]
        split_11 = None
        input_45 = torch.conv2d(
            ctx_11,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_11 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_46 = torch._C._nn.gelu(input_45, approximate="none")
        input_45 = None
        getitem_69 = gates_11[(slice(None, None, None), slice(0, 1, None))]
        mul_44 = input_46 * getitem_69
        getitem_69 = None
        ctx_all_33 = 0 + mul_44
        mul_44 = None
        input_47 = torch.conv2d(
            input_46,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_46 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_48 = torch._C._nn.gelu(input_47, approximate="none")
        input_47 = None
        getitem_70 = gates_11[(slice(None, None, None), slice(1, 2, None))]
        mul_45 = input_48 * getitem_70
        getitem_70 = None
        ctx_all_34 = ctx_all_33 + mul_45
        ctx_all_33 = mul_45 = None
        mean_11 = input_48.mean((2, 3), keepdim=True)
        input_48 = None
        ctx_global_11 = torch._C._nn.gelu(mean_11, approximate="none")
        mean_11 = None
        getitem_71 = gates_11[(slice(None, None, None), slice(2, None, None))]
        gates_11 = None
        mul_46 = ctx_global_11 * getitem_71
        ctx_global_11 = getitem_71 = None
        ctx_all_35 = ctx_all_34 + mul_46
        ctx_all_34 = mul_46 = None
        conv2d_83 = torch.conv2d(
            ctx_all_35,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_35 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_33 = q_11 * conv2d_83
        q_11 = conv2d_83 = None
        x_out_34 = torch.conv2d(
            x_out_33,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_33 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_35 = torch.nn.functional.dropout(x_out_34, 0.0, False, False)
        x_out_34 = None
        x_170 = x_165 + x_out_35
        x_165 = x_out_35 = None
        x_171 = x_170.permute(0, 2, 3, 1)
        x_172 = torch.nn.functional.layer_norm(
            x_171,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_171 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (None)
        x_173 = x_172.permute(0, 3, 1, 2)
        x_172 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_173 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_175 = torch._C._nn.gelu(x_174, approximate="none")
        x_174 = None
        x_176 = torch.nn.functional.dropout(x_175, 0.0, False, False)
        x_175 = None
        x_177 = torch.conv2d(
            x_176,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_176 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_178 = torch.nn.functional.dropout(x_177, 0.0, False, False)
        x_177 = None
        x_179 = x_170 + x_178
        x_170 = x_178 = None
        x_180 = x_179.permute(0, 2, 3, 1)
        x_181 = torch.nn.functional.layer_norm(
            x_180,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_180 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (None)
        x_182 = x_181.permute(0, 3, 1, 2)
        x_181 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_bias_ = (None)
        split_12 = torch.functional.split(x_183, [384, 384, 3], 1)
        x_183 = None
        q_12 = split_12[0]
        ctx_12 = split_12[1]
        gates_12 = split_12[2]
        split_12 = None
        input_49 = torch.conv2d(
            ctx_12,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_12 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_50 = torch._C._nn.gelu(input_49, approximate="none")
        input_49 = None
        getitem_75 = gates_12[(slice(None, None, None), slice(0, 1, None))]
        mul_48 = input_50 * getitem_75
        getitem_75 = None
        ctx_all_36 = 0 + mul_48
        mul_48 = None
        input_51 = torch.conv2d(
            input_50,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_50 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_52 = torch._C._nn.gelu(input_51, approximate="none")
        input_51 = None
        getitem_76 = gates_12[(slice(None, None, None), slice(1, 2, None))]
        mul_49 = input_52 * getitem_76
        getitem_76 = None
        ctx_all_37 = ctx_all_36 + mul_49
        ctx_all_36 = mul_49 = None
        mean_12 = input_52.mean((2, 3), keepdim=True)
        input_52 = None
        ctx_global_12 = torch._C._nn.gelu(mean_12, approximate="none")
        mean_12 = None
        getitem_77 = gates_12[(slice(None, None, None), slice(2, None, None))]
        gates_12 = None
        mul_50 = ctx_global_12 * getitem_77
        ctx_global_12 = getitem_77 = None
        ctx_all_38 = ctx_all_37 + mul_50
        ctx_all_37 = mul_50 = None
        conv2d_90 = torch.conv2d(
            ctx_all_38,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_38 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_36 = q_12 * conv2d_90
        q_12 = conv2d_90 = None
        x_out_37 = torch.conv2d(
            x_out_36,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_36 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_38 = torch.nn.functional.dropout(x_out_37, 0.0, False, False)
        x_out_37 = None
        x_184 = x_179 + x_out_38
        x_179 = x_out_38 = None
        x_185 = x_184.permute(0, 2, 3, 1)
        x_186 = torch.nn.functional.layer_norm(
            x_185,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_185 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (None)
        x_187 = x_186.permute(0, 3, 1, 2)
        x_186 = None
        x_188 = torch.conv2d(
            x_187,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_187 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_189 = torch._C._nn.gelu(x_188, approximate="none")
        x_188 = None
        x_190 = torch.nn.functional.dropout(x_189, 0.0, False, False)
        x_189 = None
        x_191 = torch.conv2d(
            x_190,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_190 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_192 = torch.nn.functional.dropout(x_191, 0.0, False, False)
        x_191 = None
        x_193 = x_184 + x_192
        x_184 = x_192 = None
        x_194 = x_193.permute(0, 2, 3, 1)
        x_195 = torch.nn.functional.layer_norm(
            x_194,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_194 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (None)
        x_196 = x_195.permute(0, 3, 1, 2)
        x_195 = None
        x_197 = torch.conv2d(
            x_196,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_196 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_bias_ = (None)
        split_13 = torch.functional.split(x_197, [384, 384, 3], 1)
        x_197 = None
        q_13 = split_13[0]
        ctx_13 = split_13[1]
        gates_13 = split_13[2]
        split_13 = None
        input_53 = torch.conv2d(
            ctx_13,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_13 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_54 = torch._C._nn.gelu(input_53, approximate="none")
        input_53 = None
        getitem_81 = gates_13[(slice(None, None, None), slice(0, 1, None))]
        mul_52 = input_54 * getitem_81
        getitem_81 = None
        ctx_all_39 = 0 + mul_52
        mul_52 = None
        input_55 = torch.conv2d(
            input_54,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_54 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_56 = torch._C._nn.gelu(input_55, approximate="none")
        input_55 = None
        getitem_82 = gates_13[(slice(None, None, None), slice(1, 2, None))]
        mul_53 = input_56 * getitem_82
        getitem_82 = None
        ctx_all_40 = ctx_all_39 + mul_53
        ctx_all_39 = mul_53 = None
        mean_13 = input_56.mean((2, 3), keepdim=True)
        input_56 = None
        ctx_global_13 = torch._C._nn.gelu(mean_13, approximate="none")
        mean_13 = None
        getitem_83 = gates_13[(slice(None, None, None), slice(2, None, None))]
        gates_13 = None
        mul_54 = ctx_global_13 * getitem_83
        ctx_global_13 = getitem_83 = None
        ctx_all_41 = ctx_all_40 + mul_54
        ctx_all_40 = mul_54 = None
        conv2d_97 = torch.conv2d(
            ctx_all_41,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_41 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_39 = q_13 * conv2d_97
        q_13 = conv2d_97 = None
        x_out_40 = torch.conv2d(
            x_out_39,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_39 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_41 = torch.nn.functional.dropout(x_out_40, 0.0, False, False)
        x_out_40 = None
        x_198 = x_193 + x_out_41
        x_193 = x_out_41 = None
        x_199 = x_198.permute(0, 2, 3, 1)
        x_200 = torch.nn.functional.layer_norm(
            x_199,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_199 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (None)
        x_201 = x_200.permute(0, 3, 1, 2)
        x_200 = None
        x_202 = torch.conv2d(
            x_201,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_201 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_203 = torch._C._nn.gelu(x_202, approximate="none")
        x_202 = None
        x_204 = torch.nn.functional.dropout(x_203, 0.0, False, False)
        x_203 = None
        x_205 = torch.conv2d(
            x_204,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_204 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_206 = torch.nn.functional.dropout(x_205, 0.0, False, False)
        x_205 = None
        x_207 = x_198 + x_206
        x_198 = x_206 = None
        x_208 = x_207.permute(0, 2, 3, 1)
        x_209 = torch.nn.functional.layer_norm(
            x_208,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_208 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (None)
        x_210 = x_209.permute(0, 3, 1, 2)
        x_209 = None
        x_211 = torch.conv2d(
            x_210,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_210 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_bias_ = (None)
        split_14 = torch.functional.split(x_211, [384, 384, 3], 1)
        x_211 = None
        q_14 = split_14[0]
        ctx_14 = split_14[1]
        gates_14 = split_14[2]
        split_14 = None
        input_57 = torch.conv2d(
            ctx_14,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_14 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_58 = torch._C._nn.gelu(input_57, approximate="none")
        input_57 = None
        getitem_87 = gates_14[(slice(None, None, None), slice(0, 1, None))]
        mul_56 = input_58 * getitem_87
        getitem_87 = None
        ctx_all_42 = 0 + mul_56
        mul_56 = None
        input_59 = torch.conv2d(
            input_58,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_58 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_60 = torch._C._nn.gelu(input_59, approximate="none")
        input_59 = None
        getitem_88 = gates_14[(slice(None, None, None), slice(1, 2, None))]
        mul_57 = input_60 * getitem_88
        getitem_88 = None
        ctx_all_43 = ctx_all_42 + mul_57
        ctx_all_42 = mul_57 = None
        mean_14 = input_60.mean((2, 3), keepdim=True)
        input_60 = None
        ctx_global_14 = torch._C._nn.gelu(mean_14, approximate="none")
        mean_14 = None
        getitem_89 = gates_14[(slice(None, None, None), slice(2, None, None))]
        gates_14 = None
        mul_58 = ctx_global_14 * getitem_89
        ctx_global_14 = getitem_89 = None
        ctx_all_44 = ctx_all_43 + mul_58
        ctx_all_43 = mul_58 = None
        conv2d_104 = torch.conv2d(
            ctx_all_44,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_44 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_42 = q_14 * conv2d_104
        q_14 = conv2d_104 = None
        x_out_43 = torch.conv2d(
            x_out_42,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_42 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_44 = torch.nn.functional.dropout(x_out_43, 0.0, False, False)
        x_out_43 = None
        x_212 = x_207 + x_out_44
        x_207 = x_out_44 = None
        x_213 = x_212.permute(0, 2, 3, 1)
        x_214 = torch.nn.functional.layer_norm(
            x_213,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_213 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (None)
        x_215 = x_214.permute(0, 3, 1, 2)
        x_214 = None
        x_216 = torch.conv2d(
            x_215,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_215 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_217 = torch._C._nn.gelu(x_216, approximate="none")
        x_216 = None
        x_218 = torch.nn.functional.dropout(x_217, 0.0, False, False)
        x_217 = None
        x_219 = torch.conv2d(
            x_218,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_218 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_220 = torch.nn.functional.dropout(x_219, 0.0, False, False)
        x_219 = None
        x_221 = x_212 + x_220
        x_212 = x_220 = None
        x_222 = x_221.permute(0, 2, 3, 1)
        x_223 = torch.nn.functional.layer_norm(
            x_222,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_222 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (None)
        x_224 = x_223.permute(0, 3, 1, 2)
        x_223 = None
        x_225 = torch.conv2d(
            x_224,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_224 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_bias_ = (None)
        split_15 = torch.functional.split(x_225, [384, 384, 3], 1)
        x_225 = None
        q_15 = split_15[0]
        ctx_15 = split_15[1]
        gates_15 = split_15[2]
        split_15 = None
        input_61 = torch.conv2d(
            ctx_15,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_15 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_62 = torch._C._nn.gelu(input_61, approximate="none")
        input_61 = None
        getitem_93 = gates_15[(slice(None, None, None), slice(0, 1, None))]
        mul_60 = input_62 * getitem_93
        getitem_93 = None
        ctx_all_45 = 0 + mul_60
        mul_60 = None
        input_63 = torch.conv2d(
            input_62,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_62 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_64 = torch._C._nn.gelu(input_63, approximate="none")
        input_63 = None
        getitem_94 = gates_15[(slice(None, None, None), slice(1, 2, None))]
        mul_61 = input_64 * getitem_94
        getitem_94 = None
        ctx_all_46 = ctx_all_45 + mul_61
        ctx_all_45 = mul_61 = None
        mean_15 = input_64.mean((2, 3), keepdim=True)
        input_64 = None
        ctx_global_15 = torch._C._nn.gelu(mean_15, approximate="none")
        mean_15 = None
        getitem_95 = gates_15[(slice(None, None, None), slice(2, None, None))]
        gates_15 = None
        mul_62 = ctx_global_15 * getitem_95
        ctx_global_15 = getitem_95 = None
        ctx_all_47 = ctx_all_46 + mul_62
        ctx_all_46 = mul_62 = None
        conv2d_111 = torch.conv2d(
            ctx_all_47,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_47 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_45 = q_15 * conv2d_111
        q_15 = conv2d_111 = None
        x_out_46 = torch.conv2d(
            x_out_45,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_45 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_47 = torch.nn.functional.dropout(x_out_46, 0.0, False, False)
        x_out_46 = None
        x_226 = x_221 + x_out_47
        x_221 = x_out_47 = None
        x_227 = x_226.permute(0, 2, 3, 1)
        x_228 = torch.nn.functional.layer_norm(
            x_227,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_227 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (None)
        x_229 = x_228.permute(0, 3, 1, 2)
        x_228 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_231 = torch._C._nn.gelu(x_230, approximate="none")
        x_230 = None
        x_232 = torch.nn.functional.dropout(x_231, 0.0, False, False)
        x_231 = None
        x_233 = torch.conv2d(
            x_232,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_232 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_234 = torch.nn.functional.dropout(x_233, 0.0, False, False)
        x_233 = None
        x_235 = x_226 + x_234
        x_226 = x_234 = None
        x_236 = x_235.permute(0, 2, 3, 1)
        x_237 = torch.nn.functional.layer_norm(
            x_236,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_236 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_ = (None)
        x_238 = x_237.permute(0, 3, 1, 2)
        x_237 = None
        x_239 = torch.conv2d(
            x_238,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_238 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_bias_ = (None)
        split_16 = torch.functional.split(x_239, [384, 384, 3], 1)
        x_239 = None
        q_16 = split_16[0]
        ctx_16 = split_16[1]
        gates_16 = split_16[2]
        split_16 = None
        input_65 = torch.conv2d(
            ctx_16,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_16 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_66 = torch._C._nn.gelu(input_65, approximate="none")
        input_65 = None
        getitem_99 = gates_16[(slice(None, None, None), slice(0, 1, None))]
        mul_64 = input_66 * getitem_99
        getitem_99 = None
        ctx_all_48 = 0 + mul_64
        mul_64 = None
        input_67 = torch.conv2d(
            input_66,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_66 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_68 = torch._C._nn.gelu(input_67, approximate="none")
        input_67 = None
        getitem_100 = gates_16[(slice(None, None, None), slice(1, 2, None))]
        mul_65 = input_68 * getitem_100
        getitem_100 = None
        ctx_all_49 = ctx_all_48 + mul_65
        ctx_all_48 = mul_65 = None
        mean_16 = input_68.mean((2, 3), keepdim=True)
        input_68 = None
        ctx_global_16 = torch._C._nn.gelu(mean_16, approximate="none")
        mean_16 = None
        getitem_101 = gates_16[(slice(None, None, None), slice(2, None, None))]
        gates_16 = None
        mul_66 = ctx_global_16 * getitem_101
        ctx_global_16 = getitem_101 = None
        ctx_all_50 = ctx_all_49 + mul_66
        ctx_all_49 = mul_66 = None
        conv2d_118 = torch.conv2d(
            ctx_all_50,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_50 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_48 = q_16 * conv2d_118
        q_16 = conv2d_118 = None
        x_out_49 = torch.conv2d(
            x_out_48,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_48 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_50 = torch.nn.functional.dropout(x_out_49, 0.0, False, False)
        x_out_49 = None
        x_240 = x_235 + x_out_50
        x_235 = x_out_50 = None
        x_241 = x_240.permute(0, 2, 3, 1)
        x_242 = torch.nn.functional.layer_norm(
            x_241,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_241 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_ = (None)
        x_243 = x_242.permute(0, 3, 1, 2)
        x_242 = None
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_243 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_245 = torch._C._nn.gelu(x_244, approximate="none")
        x_244 = None
        x_246 = torch.nn.functional.dropout(x_245, 0.0, False, False)
        x_245 = None
        x_247 = torch.conv2d(
            x_246,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_246 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_248 = torch.nn.functional.dropout(x_247, 0.0, False, False)
        x_247 = None
        x_249 = x_240 + x_248
        x_240 = x_248 = None
        x_250 = x_249.permute(0, 2, 3, 1)
        x_251 = torch.nn.functional.layer_norm(
            x_250,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_250 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_ = (None)
        x_252 = x_251.permute(0, 3, 1, 2)
        x_251 = None
        x_253 = torch.conv2d(
            x_252,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_252 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_bias_ = (None)
        split_17 = torch.functional.split(x_253, [384, 384, 3], 1)
        x_253 = None
        q_17 = split_17[0]
        ctx_17 = split_17[1]
        gates_17 = split_17[2]
        split_17 = None
        input_69 = torch.conv2d(
            ctx_17,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_17 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_70 = torch._C._nn.gelu(input_69, approximate="none")
        input_69 = None
        getitem_105 = gates_17[(slice(None, None, None), slice(0, 1, None))]
        mul_68 = input_70 * getitem_105
        getitem_105 = None
        ctx_all_51 = 0 + mul_68
        mul_68 = None
        input_71 = torch.conv2d(
            input_70,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_70 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_72 = torch._C._nn.gelu(input_71, approximate="none")
        input_71 = None
        getitem_106 = gates_17[(slice(None, None, None), slice(1, 2, None))]
        mul_69 = input_72 * getitem_106
        getitem_106 = None
        ctx_all_52 = ctx_all_51 + mul_69
        ctx_all_51 = mul_69 = None
        mean_17 = input_72.mean((2, 3), keepdim=True)
        input_72 = None
        ctx_global_17 = torch._C._nn.gelu(mean_17, approximate="none")
        mean_17 = None
        getitem_107 = gates_17[(slice(None, None, None), slice(2, None, None))]
        gates_17 = None
        mul_70 = ctx_global_17 * getitem_107
        ctx_global_17 = getitem_107 = None
        ctx_all_53 = ctx_all_52 + mul_70
        ctx_all_52 = mul_70 = None
        conv2d_125 = torch.conv2d(
            ctx_all_53,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_53 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_51 = q_17 * conv2d_125
        q_17 = conv2d_125 = None
        x_out_52 = torch.conv2d(
            x_out_51,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_51 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_53 = torch.nn.functional.dropout(x_out_52, 0.0, False, False)
        x_out_52 = None
        x_254 = x_249 + x_out_53
        x_249 = x_out_53 = None
        x_255 = x_254.permute(0, 2, 3, 1)
        x_256 = torch.nn.functional.layer_norm(
            x_255,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_255 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_ = (None)
        x_257 = x_256.permute(0, 3, 1, 2)
        x_256 = None
        x_258 = torch.conv2d(
            x_257,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_257 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_259 = torch._C._nn.gelu(x_258, approximate="none")
        x_258 = None
        x_260 = torch.nn.functional.dropout(x_259, 0.0, False, False)
        x_259 = None
        x_261 = torch.conv2d(
            x_260,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_260 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_262 = torch.nn.functional.dropout(x_261, 0.0, False, False)
        x_261 = None
        x_263 = x_254 + x_262
        x_254 = x_262 = None
        x_264 = x_263.permute(0, 2, 3, 1)
        x_265 = torch.nn.functional.layer_norm(
            x_264,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_264 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_ = (None)
        x_266 = x_265.permute(0, 3, 1, 2)
        x_265 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_266 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_bias_ = (None)
        split_18 = torch.functional.split(x_267, [384, 384, 3], 1)
        x_267 = None
        q_18 = split_18[0]
        ctx_18 = split_18[1]
        gates_18 = split_18[2]
        split_18 = None
        input_73 = torch.conv2d(
            ctx_18,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_18 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_74 = torch._C._nn.gelu(input_73, approximate="none")
        input_73 = None
        getitem_111 = gates_18[(slice(None, None, None), slice(0, 1, None))]
        mul_72 = input_74 * getitem_111
        getitem_111 = None
        ctx_all_54 = 0 + mul_72
        mul_72 = None
        input_75 = torch.conv2d(
            input_74,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_74 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_76 = torch._C._nn.gelu(input_75, approximate="none")
        input_75 = None
        getitem_112 = gates_18[(slice(None, None, None), slice(1, 2, None))]
        mul_73 = input_76 * getitem_112
        getitem_112 = None
        ctx_all_55 = ctx_all_54 + mul_73
        ctx_all_54 = mul_73 = None
        mean_18 = input_76.mean((2, 3), keepdim=True)
        input_76 = None
        ctx_global_18 = torch._C._nn.gelu(mean_18, approximate="none")
        mean_18 = None
        getitem_113 = gates_18[(slice(None, None, None), slice(2, None, None))]
        gates_18 = None
        mul_74 = ctx_global_18 * getitem_113
        ctx_global_18 = getitem_113 = None
        ctx_all_56 = ctx_all_55 + mul_74
        ctx_all_55 = mul_74 = None
        conv2d_132 = torch.conv2d(
            ctx_all_56,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_56 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_54 = q_18 * conv2d_132
        q_18 = conv2d_132 = None
        x_out_55 = torch.conv2d(
            x_out_54,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_54 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_56 = torch.nn.functional.dropout(x_out_55, 0.0, False, False)
        x_out_55 = None
        x_268 = x_263 + x_out_56
        x_263 = x_out_56 = None
        x_269 = x_268.permute(0, 2, 3, 1)
        x_270 = torch.nn.functional.layer_norm(
            x_269,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_269 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_ = (None)
        x_271 = x_270.permute(0, 3, 1, 2)
        x_270 = None
        x_272 = torch.conv2d(
            x_271,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_271 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_273 = torch._C._nn.gelu(x_272, approximate="none")
        x_272 = None
        x_274 = torch.nn.functional.dropout(x_273, 0.0, False, False)
        x_273 = None
        x_275 = torch.conv2d(
            x_274,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_274 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_276 = torch.nn.functional.dropout(x_275, 0.0, False, False)
        x_275 = None
        x_277 = x_268 + x_276
        x_268 = x_276 = None
        x_278 = x_277.permute(0, 2, 3, 1)
        x_279 = torch.nn.functional.layer_norm(
            x_278,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_278 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_ = (None)
        x_280 = x_279.permute(0, 3, 1, 2)
        x_279 = None
        x_281 = torch.conv2d(
            x_280,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_280 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_bias_ = (None)
        split_19 = torch.functional.split(x_281, [384, 384, 3], 1)
        x_281 = None
        q_19 = split_19[0]
        ctx_19 = split_19[1]
        gates_19 = split_19[2]
        split_19 = None
        input_77 = torch.conv2d(
            ctx_19,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_19 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_78 = torch._C._nn.gelu(input_77, approximate="none")
        input_77 = None
        getitem_117 = gates_19[(slice(None, None, None), slice(0, 1, None))]
        mul_76 = input_78 * getitem_117
        getitem_117 = None
        ctx_all_57 = 0 + mul_76
        mul_76 = None
        input_79 = torch.conv2d(
            input_78,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_78 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_80 = torch._C._nn.gelu(input_79, approximate="none")
        input_79 = None
        getitem_118 = gates_19[(slice(None, None, None), slice(1, 2, None))]
        mul_77 = input_80 * getitem_118
        getitem_118 = None
        ctx_all_58 = ctx_all_57 + mul_77
        ctx_all_57 = mul_77 = None
        mean_19 = input_80.mean((2, 3), keepdim=True)
        input_80 = None
        ctx_global_19 = torch._C._nn.gelu(mean_19, approximate="none")
        mean_19 = None
        getitem_119 = gates_19[(slice(None, None, None), slice(2, None, None))]
        gates_19 = None
        mul_78 = ctx_global_19 * getitem_119
        ctx_global_19 = getitem_119 = None
        ctx_all_59 = ctx_all_58 + mul_78
        ctx_all_58 = mul_78 = None
        conv2d_139 = torch.conv2d(
            ctx_all_59,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_59 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_57 = q_19 * conv2d_139
        q_19 = conv2d_139 = None
        x_out_58 = torch.conv2d(
            x_out_57,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_57 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_59 = torch.nn.functional.dropout(x_out_58, 0.0, False, False)
        x_out_58 = None
        x_282 = x_277 + x_out_59
        x_277 = x_out_59 = None
        x_283 = x_282.permute(0, 2, 3, 1)
        x_284 = torch.nn.functional.layer_norm(
            x_283,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_283 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_ = (None)
        x_285 = x_284.permute(0, 3, 1, 2)
        x_284 = None
        x_286 = torch.conv2d(
            x_285,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_285 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_287 = torch._C._nn.gelu(x_286, approximate="none")
        x_286 = None
        x_288 = torch.nn.functional.dropout(x_287, 0.0, False, False)
        x_287 = None
        x_289 = torch.conv2d(
            x_288,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_288 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_290 = torch.nn.functional.dropout(x_289, 0.0, False, False)
        x_289 = None
        x_291 = x_282 + x_290
        x_282 = x_290 = None
        x_292 = x_291.permute(0, 2, 3, 1)
        x_293 = torch.nn.functional.layer_norm(
            x_292,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_292 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_ = (None)
        x_294 = x_293.permute(0, 3, 1, 2)
        x_293 = None
        x_295 = torch.conv2d(
            x_294,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_294 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_bias_ = (None)
        split_20 = torch.functional.split(x_295, [384, 384, 3], 1)
        x_295 = None
        q_20 = split_20[0]
        ctx_20 = split_20[1]
        gates_20 = split_20[2]
        split_20 = None
        input_81 = torch.conv2d(
            ctx_20,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_20 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_82 = torch._C._nn.gelu(input_81, approximate="none")
        input_81 = None
        getitem_123 = gates_20[(slice(None, None, None), slice(0, 1, None))]
        mul_80 = input_82 * getitem_123
        getitem_123 = None
        ctx_all_60 = 0 + mul_80
        mul_80 = None
        input_83 = torch.conv2d(
            input_82,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_82 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_84 = torch._C._nn.gelu(input_83, approximate="none")
        input_83 = None
        getitem_124 = gates_20[(slice(None, None, None), slice(1, 2, None))]
        mul_81 = input_84 * getitem_124
        getitem_124 = None
        ctx_all_61 = ctx_all_60 + mul_81
        ctx_all_60 = mul_81 = None
        mean_20 = input_84.mean((2, 3), keepdim=True)
        input_84 = None
        ctx_global_20 = torch._C._nn.gelu(mean_20, approximate="none")
        mean_20 = None
        getitem_125 = gates_20[(slice(None, None, None), slice(2, None, None))]
        gates_20 = None
        mul_82 = ctx_global_20 * getitem_125
        ctx_global_20 = getitem_125 = None
        ctx_all_62 = ctx_all_61 + mul_82
        ctx_all_61 = mul_82 = None
        conv2d_146 = torch.conv2d(
            ctx_all_62,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_62 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_60 = q_20 * conv2d_146
        q_20 = conv2d_146 = None
        x_out_61 = torch.conv2d(
            x_out_60,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_60 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_62 = torch.nn.functional.dropout(x_out_61, 0.0, False, False)
        x_out_61 = None
        x_296 = x_291 + x_out_62
        x_291 = x_out_62 = None
        x_297 = x_296.permute(0, 2, 3, 1)
        x_298 = torch.nn.functional.layer_norm(
            x_297,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_297 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_ = (None)
        x_299 = x_298.permute(0, 3, 1, 2)
        x_298 = None
        x_300 = torch.conv2d(
            x_299,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_299 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_301 = torch._C._nn.gelu(x_300, approximate="none")
        x_300 = None
        x_302 = torch.nn.functional.dropout(x_301, 0.0, False, False)
        x_301 = None
        x_303 = torch.conv2d(
            x_302,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_302 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_304 = torch.nn.functional.dropout(x_303, 0.0, False, False)
        x_303 = None
        x_305 = x_296 + x_304
        x_296 = x_304 = None
        x_306 = x_305.permute(0, 2, 3, 1)
        x_307 = torch.nn.functional.layer_norm(
            x_306,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_306 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_ = (None)
        x_308 = x_307.permute(0, 3, 1, 2)
        x_307 = None
        x_309 = torch.conv2d(
            x_308,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_308 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_bias_ = (None)
        split_21 = torch.functional.split(x_309, [384, 384, 3], 1)
        x_309 = None
        q_21 = split_21[0]
        ctx_21 = split_21[1]
        gates_21 = split_21[2]
        split_21 = None
        input_85 = torch.conv2d(
            ctx_21,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_21 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_86 = torch._C._nn.gelu(input_85, approximate="none")
        input_85 = None
        getitem_129 = gates_21[(slice(None, None, None), slice(0, 1, None))]
        mul_84 = input_86 * getitem_129
        getitem_129 = None
        ctx_all_63 = 0 + mul_84
        mul_84 = None
        input_87 = torch.conv2d(
            input_86,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_86 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_88 = torch._C._nn.gelu(input_87, approximate="none")
        input_87 = None
        getitem_130 = gates_21[(slice(None, None, None), slice(1, 2, None))]
        mul_85 = input_88 * getitem_130
        getitem_130 = None
        ctx_all_64 = ctx_all_63 + mul_85
        ctx_all_63 = mul_85 = None
        mean_21 = input_88.mean((2, 3), keepdim=True)
        input_88 = None
        ctx_global_21 = torch._C._nn.gelu(mean_21, approximate="none")
        mean_21 = None
        getitem_131 = gates_21[(slice(None, None, None), slice(2, None, None))]
        gates_21 = None
        mul_86 = ctx_global_21 * getitem_131
        ctx_global_21 = getitem_131 = None
        ctx_all_65 = ctx_all_64 + mul_86
        ctx_all_64 = mul_86 = None
        conv2d_153 = torch.conv2d(
            ctx_all_65,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_65 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_63 = q_21 * conv2d_153
        q_21 = conv2d_153 = None
        x_out_64 = torch.conv2d(
            x_out_63,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_63 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_65 = torch.nn.functional.dropout(x_out_64, 0.0, False, False)
        x_out_64 = None
        x_310 = x_305 + x_out_65
        x_305 = x_out_65 = None
        x_311 = x_310.permute(0, 2, 3, 1)
        x_312 = torch.nn.functional.layer_norm(
            x_311,
            (384,),
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_311 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_ = (None)
        x_313 = x_312.permute(0, 3, 1, 2)
        x_312 = None
        x_314 = torch.conv2d(
            x_313,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_313 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_315 = torch._C._nn.gelu(x_314, approximate="none")
        x_314 = None
        x_316 = torch.nn.functional.dropout(x_315, 0.0, False, False)
        x_315 = None
        x_317 = torch.conv2d(
            x_316,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_316 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_318 = torch.nn.functional.dropout(x_317, 0.0, False, False)
        x_317 = None
        x_319 = x_310 + x_318
        x_310 = x_318 = None
        x_320 = torch.conv2d(
            x_319,
            l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_319 = l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_ = (None)
        x_321 = x_320.permute(0, 2, 3, 1)
        x_320 = None
        x_322 = torch.nn.functional.layer_norm(
            x_321,
            (768,),
            l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_321 = l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_323 = x_322.permute(0, 3, 1, 2)
        x_322 = None
        x_324 = x_323.permute(0, 2, 3, 1)
        x_325 = torch.nn.functional.layer_norm(
            x_324,
            (768,),
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_324 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_326 = x_325.permute(0, 3, 1, 2)
        x_325 = None
        x_327 = torch.conv2d(
            x_326,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_326 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = (None)
        split_22 = torch.functional.split(x_327, [768, 768, 3], 1)
        x_327 = None
        q_22 = split_22[0]
        ctx_22 = split_22[1]
        gates_22 = split_22[2]
        split_22 = None
        input_89 = torch.conv2d(
            ctx_22,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_22 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_90 = torch._C._nn.gelu(input_89, approximate="none")
        input_89 = None
        getitem_135 = gates_22[(slice(None, None, None), slice(0, 1, None))]
        mul_88 = input_90 * getitem_135
        getitem_135 = None
        ctx_all_66 = 0 + mul_88
        mul_88 = None
        input_91 = torch.conv2d(
            input_90,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_90 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_92 = torch._C._nn.gelu(input_91, approximate="none")
        input_91 = None
        getitem_136 = gates_22[(slice(None, None, None), slice(1, 2, None))]
        mul_89 = input_92 * getitem_136
        getitem_136 = None
        ctx_all_67 = ctx_all_66 + mul_89
        ctx_all_66 = mul_89 = None
        mean_22 = input_92.mean((2, 3), keepdim=True)
        input_92 = None
        ctx_global_22 = torch._C._nn.gelu(mean_22, approximate="none")
        mean_22 = None
        getitem_137 = gates_22[(slice(None, None, None), slice(2, None, None))]
        gates_22 = None
        mul_90 = ctx_global_22 * getitem_137
        ctx_global_22 = getitem_137 = None
        ctx_all_68 = ctx_all_67 + mul_90
        ctx_all_67 = mul_90 = None
        conv2d_161 = torch.conv2d(
            ctx_all_68,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_68 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_66 = q_22 * conv2d_161
        q_22 = conv2d_161 = None
        x_out_67 = torch.conv2d(
            x_out_66,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_66 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_68 = torch.nn.functional.dropout(x_out_67, 0.0, False, False)
        x_out_67 = None
        x_328 = x_323 + x_out_68
        x_323 = x_out_68 = None
        x_329 = x_328.permute(0, 2, 3, 1)
        x_330 = torch.nn.functional.layer_norm(
            x_329,
            (768,),
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_329 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_331 = x_330.permute(0, 3, 1, 2)
        x_330 = None
        x_332 = torch.conv2d(
            x_331,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_331 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_333 = torch._C._nn.gelu(x_332, approximate="none")
        x_332 = None
        x_334 = torch.nn.functional.dropout(x_333, 0.0, False, False)
        x_333 = None
        x_335 = torch.conv2d(
            x_334,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_334 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_336 = torch.nn.functional.dropout(x_335, 0.0, False, False)
        x_335 = None
        x_337 = x_328 + x_336
        x_328 = x_336 = None
        x_338 = x_337.permute(0, 2, 3, 1)
        x_339 = torch.nn.functional.layer_norm(
            x_338,
            (768,),
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_338 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_340 = x_339.permute(0, 3, 1, 2)
        x_339 = None
        x_341 = torch.conv2d(
            x_340,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_340 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = (None)
        split_23 = torch.functional.split(x_341, [768, 768, 3], 1)
        x_341 = None
        q_23 = split_23[0]
        ctx_23 = split_23[1]
        gates_23 = split_23[2]
        split_23 = None
        input_93 = torch.conv2d(
            ctx_23,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_23 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_94 = torch._C._nn.gelu(input_93, approximate="none")
        input_93 = None
        getitem_141 = gates_23[(slice(None, None, None), slice(0, 1, None))]
        mul_92 = input_94 * getitem_141
        getitem_141 = None
        ctx_all_69 = 0 + mul_92
        mul_92 = None
        input_95 = torch.conv2d(
            input_94,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_94 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_96 = torch._C._nn.gelu(input_95, approximate="none")
        input_95 = None
        getitem_142 = gates_23[(slice(None, None, None), slice(1, 2, None))]
        mul_93 = input_96 * getitem_142
        getitem_142 = None
        ctx_all_70 = ctx_all_69 + mul_93
        ctx_all_69 = mul_93 = None
        mean_23 = input_96.mean((2, 3), keepdim=True)
        input_96 = None
        ctx_global_23 = torch._C._nn.gelu(mean_23, approximate="none")
        mean_23 = None
        getitem_143 = gates_23[(slice(None, None, None), slice(2, None, None))]
        gates_23 = None
        mul_94 = ctx_global_23 * getitem_143
        ctx_global_23 = getitem_143 = None
        ctx_all_71 = ctx_all_70 + mul_94
        ctx_all_70 = mul_94 = None
        conv2d_168 = torch.conv2d(
            ctx_all_71,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_71 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_69 = q_23 * conv2d_168
        q_23 = conv2d_168 = None
        x_out_70 = torch.conv2d(
            x_out_69,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_69 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_71 = torch.nn.functional.dropout(x_out_70, 0.0, False, False)
        x_out_70 = None
        x_342 = x_337 + x_out_71
        x_337 = x_out_71 = None
        x_343 = x_342.permute(0, 2, 3, 1)
        x_344 = torch.nn.functional.layer_norm(
            x_343,
            (768,),
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_343 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_345 = x_344.permute(0, 3, 1, 2)
        x_344 = None
        x_346 = torch.conv2d(
            x_345,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_345 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_347 = torch._C._nn.gelu(x_346, approximate="none")
        x_346 = None
        x_348 = torch.nn.functional.dropout(x_347, 0.0, False, False)
        x_347 = None
        x_349 = torch.conv2d(
            x_348,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_348 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_350 = torch.nn.functional.dropout(x_349, 0.0, False, False)
        x_349 = None
        x_351 = x_342 + x_350
        x_342 = x_350 = None
        x_352 = x_351.permute(0, 2, 3, 1)
        x_351 = None
        x_353 = torch.nn.functional.layer_norm(
            x_352,
            (768,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-05,
        )
        x_352 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_354 = x_353.permute(0, 3, 1, 2)
        x_353 = None
        x_355 = torch.nn.functional.adaptive_avg_pool2d(x_354, 1)
        x_354 = None
        x_356 = x_355.flatten(1, -1)
        x_355 = None
        x_357 = torch.nn.functional.dropout(x_356, 0.0, False, False)
        x_356 = None
        x_358 = torch._C._nn.linear(
            x_357,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_357 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_358,)
