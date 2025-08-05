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
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
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
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
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
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
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
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
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
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
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
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
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
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
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
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
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
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
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
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
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
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
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
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
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
        split = torch.functional.split(x_7, [96, 96, 4], 1)
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
        input_5 = torch.conv2d(
            input_4,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            96,
        )
        input_4 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_6 = torch._C._nn.gelu(input_5, approximate="none")
        input_5 = None
        getitem_5 = gates[(slice(None, None, None), slice(2, 3, None))]
        mul_2 = input_6 * getitem_5
        getitem_5 = None
        ctx_all_2 = ctx_all_1 + mul_2
        ctx_all_1 = mul_2 = None
        mean = input_6.mean((2, 3), keepdim=True)
        input_6 = None
        ctx_global = torch._C._nn.gelu(mean, approximate="none")
        mean = None
        getitem_6 = gates[(slice(None, None, None), slice(3, None, None))]
        gates = None
        mul_3 = ctx_global * getitem_6
        ctx_global = getitem_6 = None
        ctx_all_3 = ctx_all_2 + mul_3
        ctx_all_2 = mul_3 = None
        conv2d_5 = torch.conv2d(
            ctx_all_3,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_3 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out = q * conv2d_5
        q = conv2d_5 = None
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
        split_1 = torch.functional.split(x_21, [96, 96, 4], 1)
        x_21 = None
        q_1 = split_1[0]
        ctx_1 = split_1[1]
        gates_1 = split_1[2]
        split_1 = None
        input_7 = torch.conv2d(
            ctx_1,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        ctx_1 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_8 = torch._C._nn.gelu(input_7, approximate="none")
        input_7 = None
        getitem_10 = gates_1[(slice(None, None, None), slice(0, 1, None))]
        mul_5 = input_8 * getitem_10
        getitem_10 = None
        ctx_all_4 = 0 + mul_5
        mul_5 = None
        input_9 = torch.conv2d(
            input_8,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        input_8 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_10 = torch._C._nn.gelu(input_9, approximate="none")
        input_9 = None
        getitem_11 = gates_1[(slice(None, None, None), slice(1, 2, None))]
        mul_6 = input_10 * getitem_11
        getitem_11 = None
        ctx_all_5 = ctx_all_4 + mul_6
        ctx_all_4 = mul_6 = None
        input_11 = torch.conv2d(
            input_10,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            96,
        )
        input_10 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_12 = torch._C._nn.gelu(input_11, approximate="none")
        input_11 = None
        getitem_12 = gates_1[(slice(None, None, None), slice(2, 3, None))]
        mul_7 = input_12 * getitem_12
        getitem_12 = None
        ctx_all_6 = ctx_all_5 + mul_7
        ctx_all_5 = mul_7 = None
        mean_1 = input_12.mean((2, 3), keepdim=True)
        input_12 = None
        ctx_global_1 = torch._C._nn.gelu(mean_1, approximate="none")
        mean_1 = None
        getitem_13 = gates_1[(slice(None, None, None), slice(3, None, None))]
        gates_1 = None
        mul_8 = ctx_global_1 * getitem_13
        ctx_global_1 = getitem_13 = None
        ctx_all_7 = ctx_all_6 + mul_8
        ctx_all_6 = mul_8 = None
        conv2d_13 = torch.conv2d(
            ctx_all_7,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_7 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_3 = q_1 * conv2d_13
        q_1 = conv2d_13 = None
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
        split_2 = torch.functional.split(x_39, [192, 192, 4], 1)
        x_39 = None
        q_2 = split_2[0]
        ctx_2 = split_2[1]
        gates_2 = split_2[2]
        split_2 = None
        input_13 = torch.conv2d(
            ctx_2,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        ctx_2 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_14 = torch._C._nn.gelu(input_13, approximate="none")
        input_13 = None
        getitem_17 = gates_2[(slice(None, None, None), slice(0, 1, None))]
        mul_10 = input_14 * getitem_17
        getitem_17 = None
        ctx_all_8 = 0 + mul_10
        mul_10 = None
        input_15 = torch.conv2d(
            input_14,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        input_14 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_16 = torch._C._nn.gelu(input_15, approximate="none")
        input_15 = None
        getitem_18 = gates_2[(slice(None, None, None), slice(1, 2, None))]
        mul_11 = input_16 * getitem_18
        getitem_18 = None
        ctx_all_9 = ctx_all_8 + mul_11
        ctx_all_8 = mul_11 = None
        input_17 = torch.conv2d(
            input_16,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        input_16 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_18 = torch._C._nn.gelu(input_17, approximate="none")
        input_17 = None
        getitem_19 = gates_2[(slice(None, None, None), slice(2, 3, None))]
        mul_12 = input_18 * getitem_19
        getitem_19 = None
        ctx_all_10 = ctx_all_9 + mul_12
        ctx_all_9 = mul_12 = None
        mean_2 = input_18.mean((2, 3), keepdim=True)
        input_18 = None
        ctx_global_2 = torch._C._nn.gelu(mean_2, approximate="none")
        mean_2 = None
        getitem_20 = gates_2[(slice(None, None, None), slice(3, None, None))]
        gates_2 = None
        mul_13 = ctx_global_2 * getitem_20
        ctx_global_2 = getitem_20 = None
        ctx_all_11 = ctx_all_10 + mul_13
        ctx_all_10 = mul_13 = None
        conv2d_22 = torch.conv2d(
            ctx_all_11,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_11 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_6 = q_2 * conv2d_22
        q_2 = conv2d_22 = None
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
        split_3 = torch.functional.split(x_53, [192, 192, 4], 1)
        x_53 = None
        q_3 = split_3[0]
        ctx_3 = split_3[1]
        gates_3 = split_3[2]
        split_3 = None
        input_19 = torch.conv2d(
            ctx_3,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        ctx_3 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_20 = torch._C._nn.gelu(input_19, approximate="none")
        input_19 = None
        getitem_24 = gates_3[(slice(None, None, None), slice(0, 1, None))]
        mul_15 = input_20 * getitem_24
        getitem_24 = None
        ctx_all_12 = 0 + mul_15
        mul_15 = None
        input_21 = torch.conv2d(
            input_20,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        input_20 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_22 = torch._C._nn.gelu(input_21, approximate="none")
        input_21 = None
        getitem_25 = gates_3[(slice(None, None, None), slice(1, 2, None))]
        mul_16 = input_22 * getitem_25
        getitem_25 = None
        ctx_all_13 = ctx_all_12 + mul_16
        ctx_all_12 = mul_16 = None
        input_23 = torch.conv2d(
            input_22,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        input_22 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_24 = torch._C._nn.gelu(input_23, approximate="none")
        input_23 = None
        getitem_26 = gates_3[(slice(None, None, None), slice(2, 3, None))]
        mul_17 = input_24 * getitem_26
        getitem_26 = None
        ctx_all_14 = ctx_all_13 + mul_17
        ctx_all_13 = mul_17 = None
        mean_3 = input_24.mean((2, 3), keepdim=True)
        input_24 = None
        ctx_global_3 = torch._C._nn.gelu(mean_3, approximate="none")
        mean_3 = None
        getitem_27 = gates_3[(slice(None, None, None), slice(3, None, None))]
        gates_3 = None
        mul_18 = ctx_global_3 * getitem_27
        ctx_global_3 = getitem_27 = None
        ctx_all_15 = ctx_all_14 + mul_18
        ctx_all_14 = mul_18 = None
        conv2d_30 = torch.conv2d(
            ctx_all_15,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_15 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_9 = q_3 * conv2d_30
        q_3 = conv2d_30 = None
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
        split_4 = torch.functional.split(x_71, [384, 384, 4], 1)
        x_71 = None
        q_4 = split_4[0]
        ctx_4 = split_4[1]
        gates_4 = split_4[2]
        split_4 = None
        input_25 = torch.conv2d(
            ctx_4,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_4 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_26 = torch._C._nn.gelu(input_25, approximate="none")
        input_25 = None
        getitem_31 = gates_4[(slice(None, None, None), slice(0, 1, None))]
        mul_20 = input_26 * getitem_31
        getitem_31 = None
        ctx_all_16 = 0 + mul_20
        mul_20 = None
        input_27 = torch.conv2d(
            input_26,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_26 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_28 = torch._C._nn.gelu(input_27, approximate="none")
        input_27 = None
        getitem_32 = gates_4[(slice(None, None, None), slice(1, 2, None))]
        mul_21 = input_28 * getitem_32
        getitem_32 = None
        ctx_all_17 = ctx_all_16 + mul_21
        ctx_all_16 = mul_21 = None
        input_29 = torch.conv2d(
            input_28,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        input_28 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_30 = torch._C._nn.gelu(input_29, approximate="none")
        input_29 = None
        getitem_33 = gates_4[(slice(None, None, None), slice(2, 3, None))]
        mul_22 = input_30 * getitem_33
        getitem_33 = None
        ctx_all_18 = ctx_all_17 + mul_22
        ctx_all_17 = mul_22 = None
        mean_4 = input_30.mean((2, 3), keepdim=True)
        input_30 = None
        ctx_global_4 = torch._C._nn.gelu(mean_4, approximate="none")
        mean_4 = None
        getitem_34 = gates_4[(slice(None, None, None), slice(3, None, None))]
        gates_4 = None
        mul_23 = ctx_global_4 * getitem_34
        ctx_global_4 = getitem_34 = None
        ctx_all_19 = ctx_all_18 + mul_23
        ctx_all_18 = mul_23 = None
        conv2d_39 = torch.conv2d(
            ctx_all_19,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_19 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_12 = q_4 * conv2d_39
        q_4 = conv2d_39 = None
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
        split_5 = torch.functional.split(x_85, [384, 384, 4], 1)
        x_85 = None
        q_5 = split_5[0]
        ctx_5 = split_5[1]
        gates_5 = split_5[2]
        split_5 = None
        input_31 = torch.conv2d(
            ctx_5,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_5 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_32 = torch._C._nn.gelu(input_31, approximate="none")
        input_31 = None
        getitem_38 = gates_5[(slice(None, None, None), slice(0, 1, None))]
        mul_25 = input_32 * getitem_38
        getitem_38 = None
        ctx_all_20 = 0 + mul_25
        mul_25 = None
        input_33 = torch.conv2d(
            input_32,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_32 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_34 = torch._C._nn.gelu(input_33, approximate="none")
        input_33 = None
        getitem_39 = gates_5[(slice(None, None, None), slice(1, 2, None))]
        mul_26 = input_34 * getitem_39
        getitem_39 = None
        ctx_all_21 = ctx_all_20 + mul_26
        ctx_all_20 = mul_26 = None
        input_35 = torch.conv2d(
            input_34,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        input_34 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_36 = torch._C._nn.gelu(input_35, approximate="none")
        input_35 = None
        getitem_40 = gates_5[(slice(None, None, None), slice(2, 3, None))]
        mul_27 = input_36 * getitem_40
        getitem_40 = None
        ctx_all_22 = ctx_all_21 + mul_27
        ctx_all_21 = mul_27 = None
        mean_5 = input_36.mean((2, 3), keepdim=True)
        input_36 = None
        ctx_global_5 = torch._C._nn.gelu(mean_5, approximate="none")
        mean_5 = None
        getitem_41 = gates_5[(slice(None, None, None), slice(3, None, None))]
        gates_5 = None
        mul_28 = ctx_global_5 * getitem_41
        ctx_global_5 = getitem_41 = None
        ctx_all_23 = ctx_all_22 + mul_28
        ctx_all_22 = mul_28 = None
        conv2d_47 = torch.conv2d(
            ctx_all_23,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_23 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_15 = q_5 * conv2d_47
        q_5 = conv2d_47 = None
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
        split_6 = torch.functional.split(x_99, [384, 384, 4], 1)
        x_99 = None
        q_6 = split_6[0]
        ctx_6 = split_6[1]
        gates_6 = split_6[2]
        split_6 = None
        input_37 = torch.conv2d(
            ctx_6,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_6 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_38 = torch._C._nn.gelu(input_37, approximate="none")
        input_37 = None
        getitem_45 = gates_6[(slice(None, None, None), slice(0, 1, None))]
        mul_30 = input_38 * getitem_45
        getitem_45 = None
        ctx_all_24 = 0 + mul_30
        mul_30 = None
        input_39 = torch.conv2d(
            input_38,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_38 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_40 = torch._C._nn.gelu(input_39, approximate="none")
        input_39 = None
        getitem_46 = gates_6[(slice(None, None, None), slice(1, 2, None))]
        mul_31 = input_40 * getitem_46
        getitem_46 = None
        ctx_all_25 = ctx_all_24 + mul_31
        ctx_all_24 = mul_31 = None
        input_41 = torch.conv2d(
            input_40,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        input_40 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_42 = torch._C._nn.gelu(input_41, approximate="none")
        input_41 = None
        getitem_47 = gates_6[(slice(None, None, None), slice(2, 3, None))]
        mul_32 = input_42 * getitem_47
        getitem_47 = None
        ctx_all_26 = ctx_all_25 + mul_32
        ctx_all_25 = mul_32 = None
        mean_6 = input_42.mean((2, 3), keepdim=True)
        input_42 = None
        ctx_global_6 = torch._C._nn.gelu(mean_6, approximate="none")
        mean_6 = None
        getitem_48 = gates_6[(slice(None, None, None), slice(3, None, None))]
        gates_6 = None
        mul_33 = ctx_global_6 * getitem_48
        ctx_global_6 = getitem_48 = None
        ctx_all_27 = ctx_all_26 + mul_33
        ctx_all_26 = mul_33 = None
        conv2d_55 = torch.conv2d(
            ctx_all_27,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_27 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_18 = q_6 * conv2d_55
        q_6 = conv2d_55 = None
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
        split_7 = torch.functional.split(x_113, [384, 384, 4], 1)
        x_113 = None
        q_7 = split_7[0]
        ctx_7 = split_7[1]
        gates_7 = split_7[2]
        split_7 = None
        input_43 = torch.conv2d(
            ctx_7,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_7 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_44 = torch._C._nn.gelu(input_43, approximate="none")
        input_43 = None
        getitem_52 = gates_7[(slice(None, None, None), slice(0, 1, None))]
        mul_35 = input_44 * getitem_52
        getitem_52 = None
        ctx_all_28 = 0 + mul_35
        mul_35 = None
        input_45 = torch.conv2d(
            input_44,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_44 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_46 = torch._C._nn.gelu(input_45, approximate="none")
        input_45 = None
        getitem_53 = gates_7[(slice(None, None, None), slice(1, 2, None))]
        mul_36 = input_46 * getitem_53
        getitem_53 = None
        ctx_all_29 = ctx_all_28 + mul_36
        ctx_all_28 = mul_36 = None
        input_47 = torch.conv2d(
            input_46,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        input_46 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_48 = torch._C._nn.gelu(input_47, approximate="none")
        input_47 = None
        getitem_54 = gates_7[(slice(None, None, None), slice(2, 3, None))]
        mul_37 = input_48 * getitem_54
        getitem_54 = None
        ctx_all_30 = ctx_all_29 + mul_37
        ctx_all_29 = mul_37 = None
        mean_7 = input_48.mean((2, 3), keepdim=True)
        input_48 = None
        ctx_global_7 = torch._C._nn.gelu(mean_7, approximate="none")
        mean_7 = None
        getitem_55 = gates_7[(slice(None, None, None), slice(3, None, None))]
        gates_7 = None
        mul_38 = ctx_global_7 * getitem_55
        ctx_global_7 = getitem_55 = None
        ctx_all_31 = ctx_all_30 + mul_38
        ctx_all_30 = mul_38 = None
        conv2d_63 = torch.conv2d(
            ctx_all_31,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_31 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_21 = q_7 * conv2d_63
        q_7 = conv2d_63 = None
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
        split_8 = torch.functional.split(x_127, [384, 384, 4], 1)
        x_127 = None
        q_8 = split_8[0]
        ctx_8 = split_8[1]
        gates_8 = split_8[2]
        split_8 = None
        input_49 = torch.conv2d(
            ctx_8,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_8 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_50 = torch._C._nn.gelu(input_49, approximate="none")
        input_49 = None
        getitem_59 = gates_8[(slice(None, None, None), slice(0, 1, None))]
        mul_40 = input_50 * getitem_59
        getitem_59 = None
        ctx_all_32 = 0 + mul_40
        mul_40 = None
        input_51 = torch.conv2d(
            input_50,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_50 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_52 = torch._C._nn.gelu(input_51, approximate="none")
        input_51 = None
        getitem_60 = gates_8[(slice(None, None, None), slice(1, 2, None))]
        mul_41 = input_52 * getitem_60
        getitem_60 = None
        ctx_all_33 = ctx_all_32 + mul_41
        ctx_all_32 = mul_41 = None
        input_53 = torch.conv2d(
            input_52,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        input_52 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_54 = torch._C._nn.gelu(input_53, approximate="none")
        input_53 = None
        getitem_61 = gates_8[(slice(None, None, None), slice(2, 3, None))]
        mul_42 = input_54 * getitem_61
        getitem_61 = None
        ctx_all_34 = ctx_all_33 + mul_42
        ctx_all_33 = mul_42 = None
        mean_8 = input_54.mean((2, 3), keepdim=True)
        input_54 = None
        ctx_global_8 = torch._C._nn.gelu(mean_8, approximate="none")
        mean_8 = None
        getitem_62 = gates_8[(slice(None, None, None), slice(3, None, None))]
        gates_8 = None
        mul_43 = ctx_global_8 * getitem_62
        ctx_global_8 = getitem_62 = None
        ctx_all_35 = ctx_all_34 + mul_43
        ctx_all_34 = mul_43 = None
        conv2d_71 = torch.conv2d(
            ctx_all_35,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_35 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_24 = q_8 * conv2d_71
        q_8 = conv2d_71 = None
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
        split_9 = torch.functional.split(x_141, [384, 384, 4], 1)
        x_141 = None
        q_9 = split_9[0]
        ctx_9 = split_9[1]
        gates_9 = split_9[2]
        split_9 = None
        input_55 = torch.conv2d(
            ctx_9,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_9 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_56 = torch._C._nn.gelu(input_55, approximate="none")
        input_55 = None
        getitem_66 = gates_9[(slice(None, None, None), slice(0, 1, None))]
        mul_45 = input_56 * getitem_66
        getitem_66 = None
        ctx_all_36 = 0 + mul_45
        mul_45 = None
        input_57 = torch.conv2d(
            input_56,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_56 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_58 = torch._C._nn.gelu(input_57, approximate="none")
        input_57 = None
        getitem_67 = gates_9[(slice(None, None, None), slice(1, 2, None))]
        mul_46 = input_58 * getitem_67
        getitem_67 = None
        ctx_all_37 = ctx_all_36 + mul_46
        ctx_all_36 = mul_46 = None
        input_59 = torch.conv2d(
            input_58,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        input_58 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_60 = torch._C._nn.gelu(input_59, approximate="none")
        input_59 = None
        getitem_68 = gates_9[(slice(None, None, None), slice(2, 3, None))]
        mul_47 = input_60 * getitem_68
        getitem_68 = None
        ctx_all_38 = ctx_all_37 + mul_47
        ctx_all_37 = mul_47 = None
        mean_9 = input_60.mean((2, 3), keepdim=True)
        input_60 = None
        ctx_global_9 = torch._C._nn.gelu(mean_9, approximate="none")
        mean_9 = None
        getitem_69 = gates_9[(slice(None, None, None), slice(3, None, None))]
        gates_9 = None
        mul_48 = ctx_global_9 * getitem_69
        ctx_global_9 = getitem_69 = None
        ctx_all_39 = ctx_all_38 + mul_48
        ctx_all_38 = mul_48 = None
        conv2d_79 = torch.conv2d(
            ctx_all_39,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_39 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_27 = q_9 * conv2d_79
        q_9 = conv2d_79 = None
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
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_151 = l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_ = (None)
        x_153 = x_152.permute(0, 2, 3, 1)
        x_152 = None
        x_154 = torch.nn.functional.layer_norm(
            x_153,
            (768,),
            l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_153 = l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_155 = x_154.permute(0, 3, 1, 2)
        x_154 = None
        x_156 = x_155.permute(0, 2, 3, 1)
        x_157 = torch.nn.functional.layer_norm(
            x_156,
            (768,),
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_156 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_158 = x_157.permute(0, 3, 1, 2)
        x_157 = None
        x_159 = torch.conv2d(
            x_158,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_158 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = (None)
        split_10 = torch.functional.split(x_159, [768, 768, 4], 1)
        x_159 = None
        q_10 = split_10[0]
        ctx_10 = split_10[1]
        gates_10 = split_10[2]
        split_10 = None
        input_61 = torch.conv2d(
            ctx_10,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_10 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_62 = torch._C._nn.gelu(input_61, approximate="none")
        input_61 = None
        getitem_73 = gates_10[(slice(None, None, None), slice(0, 1, None))]
        mul_50 = input_62 * getitem_73
        getitem_73 = None
        ctx_all_40 = 0 + mul_50
        mul_50 = None
        input_63 = torch.conv2d(
            input_62,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_62 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_64 = torch._C._nn.gelu(input_63, approximate="none")
        input_63 = None
        getitem_74 = gates_10[(slice(None, None, None), slice(1, 2, None))]
        mul_51 = input_64 * getitem_74
        getitem_74 = None
        ctx_all_41 = ctx_all_40 + mul_51
        ctx_all_40 = mul_51 = None
        input_65 = torch.conv2d(
            input_64,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_64 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_66 = torch._C._nn.gelu(input_65, approximate="none")
        input_65 = None
        getitem_75 = gates_10[(slice(None, None, None), slice(2, 3, None))]
        mul_52 = input_66 * getitem_75
        getitem_75 = None
        ctx_all_42 = ctx_all_41 + mul_52
        ctx_all_41 = mul_52 = None
        mean_10 = input_66.mean((2, 3), keepdim=True)
        input_66 = None
        ctx_global_10 = torch._C._nn.gelu(mean_10, approximate="none")
        mean_10 = None
        getitem_76 = gates_10[(slice(None, None, None), slice(3, None, None))]
        gates_10 = None
        mul_53 = ctx_global_10 * getitem_76
        ctx_global_10 = getitem_76 = None
        ctx_all_43 = ctx_all_42 + mul_53
        ctx_all_42 = mul_53 = None
        conv2d_88 = torch.conv2d(
            ctx_all_43,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_43 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_30 = q_10 * conv2d_88
        q_10 = conv2d_88 = None
        x_out_31 = torch.conv2d(
            x_out_30,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_30 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_32 = torch.nn.functional.dropout(x_out_31, 0.0, False, False)
        x_out_31 = None
        x_160 = x_155 + x_out_32
        x_155 = x_out_32 = None
        x_161 = x_160.permute(0, 2, 3, 1)
        x_162 = torch.nn.functional.layer_norm(
            x_161,
            (768,),
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_161 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_163 = x_162.permute(0, 3, 1, 2)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_165 = torch._C._nn.gelu(x_164, approximate="none")
        x_164 = None
        x_166 = torch.nn.functional.dropout(x_165, 0.0, False, False)
        x_165 = None
        x_167 = torch.conv2d(
            x_166,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_166 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_168 = torch.nn.functional.dropout(x_167, 0.0, False, False)
        x_167 = None
        x_169 = x_160 + x_168
        x_160 = x_168 = None
        x_170 = x_169.permute(0, 2, 3, 1)
        x_171 = torch.nn.functional.layer_norm(
            x_170,
            (768,),
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        x_170 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_172 = x_171.permute(0, 3, 1, 2)
        x_171 = None
        x_173 = torch.conv2d(
            x_172,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_172 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = (None)
        split_11 = torch.functional.split(x_173, [768, 768, 4], 1)
        x_173 = None
        q_11 = split_11[0]
        ctx_11 = split_11[1]
        gates_11 = split_11[2]
        split_11 = None
        input_67 = torch.conv2d(
            ctx_11,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_11 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_68 = torch._C._nn.gelu(input_67, approximate="none")
        input_67 = None
        getitem_80 = gates_11[(slice(None, None, None), slice(0, 1, None))]
        mul_55 = input_68 * getitem_80
        getitem_80 = None
        ctx_all_44 = 0 + mul_55
        mul_55 = None
        input_69 = torch.conv2d(
            input_68,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_68 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_70 = torch._C._nn.gelu(input_69, approximate="none")
        input_69 = None
        getitem_81 = gates_11[(slice(None, None, None), slice(1, 2, None))]
        mul_56 = input_70 * getitem_81
        getitem_81 = None
        ctx_all_45 = ctx_all_44 + mul_56
        ctx_all_44 = mul_56 = None
        input_71 = torch.conv2d(
            input_70,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_70 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_72 = torch._C._nn.gelu(input_71, approximate="none")
        input_71 = None
        getitem_82 = gates_11[(slice(None, None, None), slice(2, 3, None))]
        mul_57 = input_72 * getitem_82
        getitem_82 = None
        ctx_all_46 = ctx_all_45 + mul_57
        ctx_all_45 = mul_57 = None
        mean_11 = input_72.mean((2, 3), keepdim=True)
        input_72 = None
        ctx_global_11 = torch._C._nn.gelu(mean_11, approximate="none")
        mean_11 = None
        getitem_83 = gates_11[(slice(None, None, None), slice(3, None, None))]
        gates_11 = None
        mul_58 = ctx_global_11 * getitem_83
        ctx_global_11 = getitem_83 = None
        ctx_all_47 = ctx_all_46 + mul_58
        ctx_all_46 = mul_58 = None
        conv2d_96 = torch.conv2d(
            ctx_all_47,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_47 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_33 = q_11 * conv2d_96
        q_11 = conv2d_96 = None
        x_out_34 = torch.conv2d(
            x_out_33,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_33 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_ = (None)
        x_out_35 = torch.nn.functional.dropout(x_out_34, 0.0, False, False)
        x_out_34 = None
        x_174 = x_169 + x_out_35
        x_169 = x_out_35 = None
        x_175 = x_174.permute(0, 2, 3, 1)
        x_176 = torch.nn.functional.layer_norm(
            x_175,
            (768,),
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        x_175 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_177 = x_176.permute(0, 3, 1, 2)
        x_176 = None
        x_178 = torch.conv2d(
            x_177,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_177 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_179 = torch._C._nn.gelu(x_178, approximate="none")
        x_178 = None
        x_180 = torch.nn.functional.dropout(x_179, 0.0, False, False)
        x_179 = None
        x_181 = torch.conv2d(
            x_180,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_180 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_182 = torch.nn.functional.dropout(x_181, 0.0, False, False)
        x_181 = None
        x_183 = x_174 + x_182
        x_174 = x_182 = None
        x_184 = x_183.permute(0, 2, 3, 1)
        x_183 = None
        x_185 = torch.nn.functional.layer_norm(
            x_184,
            (768,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-05,
        )
        x_184 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_186 = x_185.permute(0, 3, 1, 2)
        x_185 = None
        x_187 = torch.nn.functional.adaptive_avg_pool2d(x_186, 1)
        x_186 = None
        x_188 = x_187.flatten(1, -1)
        x_187 = None
        x_189 = torch.nn.functional.dropout(x_188, 0.0, False, False)
        x_188 = None
        x_190 = torch._C._nn.linear(
            x_189,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_189 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_190,)
