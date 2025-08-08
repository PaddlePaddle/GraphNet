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
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_ls1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_post_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_post_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_ls2_parameters_gamma_: torch.nn.parameter.Parameter,
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
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_ = L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_ls2_parameters_gamma_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_post_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_post_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_post_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_post_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_ls1_parameters_gamma_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_ls1_parameters_gamma_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_post_parameters_weight_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_post_parameters_weight_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_post_parameters_bias_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_post_parameters_bias_
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_ls2_parameters_gamma_ = L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_ls2_parameters_gamma_
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
            (2, 2),
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
            (192,),
            l_self_modules_stem_modules_norm_parameters_weight_,
            l_self_modules_stem_modules_norm_parameters_bias_,
            1e-05,
        )
        x_1 = (
            l_self_modules_stem_modules_norm_parameters_weight_
        ) = l_self_modules_stem_modules_norm_parameters_bias_ = None
        x_3 = x_2.permute(0, 3, 1, 2)
        x_2 = None
        x_4 = torch.conv2d(
            x_3,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = (None)
        split = torch.functional.split(x_4, [192, 192, 5], 1)
        x_4 = None
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
            192,
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
            192,
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
            192,
        )
        input_4 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_6 = torch._C._nn.gelu(input_5, approximate="none")
        input_5 = None
        getitem_5 = gates[(slice(None, None, None), slice(2, 3, None))]
        mul_2 = input_6 * getitem_5
        getitem_5 = None
        ctx_all_2 = ctx_all_1 + mul_2
        ctx_all_1 = mul_2 = None
        input_7 = torch.conv2d(
            input_6,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            192,
        )
        input_6 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_8 = torch._C._nn.gelu(input_7, approximate="none")
        input_7 = None
        getitem_6 = gates[(slice(None, None, None), slice(3, 4, None))]
        mul_3 = input_8 * getitem_6
        getitem_6 = None
        ctx_all_3 = ctx_all_2 + mul_3
        ctx_all_2 = mul_3 = None
        mean = input_8.mean((2, 3), keepdim=True)
        input_8 = None
        ctx_global = torch._C._nn.gelu(mean, approximate="none")
        mean = None
        getitem_7 = gates[(slice(None, None, None), slice(4, None, None))]
        gates = None
        mul_4 = ctx_global * getitem_7
        ctx_global = getitem_7 = None
        ctx_all_4 = ctx_all_3 + mul_4
        ctx_all_3 = mul_4 = None
        conv2d_6 = torch.conv2d(
            ctx_all_4,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_4 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out = q * conv2d_6
        q = conv2d_6 = None
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
        x_5 = x_out_2.permute(0, 2, 3, 1)
        x_out_2 = None
        x_6 = torch.nn.functional.layer_norm(
            x_5,
            (192,),
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_5 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_post_parameters_bias_ = (None)
        x_7 = x_6.permute(0, 3, 1, 2)
        x_6 = None
        gamma = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_6 = x_7 * gamma
        x_7 = gamma = None
        x_8 = x_3 + mul_6
        x_3 = mul_6 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_10 = torch._C._nn.gelu(x_9, approximate="none")
        x_9 = None
        x_11 = torch.nn.functional.dropout(x_10, 0.0, False, False)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_13 = torch.nn.functional.dropout(x_12, 0.0, False, False)
        x_12 = None
        x_14 = x_13.permute(0, 2, 3, 1)
        x_13 = None
        x_15 = torch.nn.functional.layer_norm(
            x_14,
            (192,),
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_14 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_post_parameters_bias_ = (None)
        x_16 = x_15.permute(0, 3, 1, 2)
        x_15 = None
        gamma_1 = l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_0_modules_blocks_modules_0_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_7 = x_16 * gamma_1
        x_16 = gamma_1 = None
        x_17 = x_8 + mul_7
        x_8 = mul_7 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = (None)
        split_1 = torch.functional.split(x_18, [192, 192, 5], 1)
        x_18 = None
        q_1 = split_1[0]
        ctx_1 = split_1[1]
        gates_1 = split_1[2]
        split_1 = None
        input_9 = torch.conv2d(
            ctx_1,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        ctx_1 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_10 = torch._C._nn.gelu(input_9, approximate="none")
        input_9 = None
        getitem_11 = gates_1[(slice(None, None, None), slice(0, 1, None))]
        mul_8 = input_10 * getitem_11
        getitem_11 = None
        ctx_all_5 = 0 + mul_8
        mul_8 = None
        input_11 = torch.conv2d(
            input_10,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        input_10 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_12 = torch._C._nn.gelu(input_11, approximate="none")
        input_11 = None
        getitem_12 = gates_1[(slice(None, None, None), slice(1, 2, None))]
        mul_9 = input_12 * getitem_12
        getitem_12 = None
        ctx_all_6 = ctx_all_5 + mul_9
        ctx_all_5 = mul_9 = None
        input_13 = torch.conv2d(
            input_12,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        input_12 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_14 = torch._C._nn.gelu(input_13, approximate="none")
        input_13 = None
        getitem_13 = gates_1[(slice(None, None, None), slice(2, 3, None))]
        mul_10 = input_14 * getitem_13
        getitem_13 = None
        ctx_all_7 = ctx_all_6 + mul_10
        ctx_all_6 = mul_10 = None
        input_15 = torch.conv2d(
            input_14,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            192,
        )
        input_14 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_16 = torch._C._nn.gelu(input_15, approximate="none")
        input_15 = None
        getitem_14 = gates_1[(slice(None, None, None), slice(3, 4, None))]
        mul_11 = input_16 * getitem_14
        getitem_14 = None
        ctx_all_8 = ctx_all_7 + mul_11
        ctx_all_7 = mul_11 = None
        mean_1 = input_16.mean((2, 3), keepdim=True)
        input_16 = None
        ctx_global_1 = torch._C._nn.gelu(mean_1, approximate="none")
        mean_1 = None
        getitem_15 = gates_1[(slice(None, None, None), slice(4, None, None))]
        gates_1 = None
        mul_12 = ctx_global_1 * getitem_15
        ctx_global_1 = getitem_15 = None
        ctx_all_9 = ctx_all_8 + mul_12
        ctx_all_8 = mul_12 = None
        conv2d_15 = torch.conv2d(
            ctx_all_9,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_9 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_3 = q_1 * conv2d_15
        q_1 = conv2d_15 = None
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
        x_19 = x_out_5.permute(0, 2, 3, 1)
        x_out_5 = None
        x_20 = torch.nn.functional.layer_norm(
            x_19,
            (192,),
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_19 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_post_parameters_bias_ = (None)
        x_21 = x_20.permute(0, 3, 1, 2)
        x_20 = None
        gamma_2 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_14 = x_21 * gamma_2
        x_21 = gamma_2 = None
        x_22 = x_17 + mul_14
        x_17 = mul_14 = None
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_24 = torch._C._nn.gelu(x_23, approximate="none")
        x_23 = None
        x_25 = torch.nn.functional.dropout(x_24, 0.0, False, False)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_27 = torch.nn.functional.dropout(x_26, 0.0, False, False)
        x_26 = None
        x_28 = x_27.permute(0, 2, 3, 1)
        x_27 = None
        x_29 = torch.nn.functional.layer_norm(
            x_28,
            (192,),
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_28 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_post_parameters_bias_ = (None)
        x_30 = x_29.permute(0, 3, 1, 2)
        x_29 = None
        gamma_3 = l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_0_modules_blocks_modules_1_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_15 = x_30 * gamma_3
        x_30 = gamma_3 = None
        x_31 = x_22 + mul_15
        x_22 = mul_15 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_bias_ = (None)
        x_33 = x_32.permute(0, 2, 3, 1)
        x_32 = None
        x_34 = torch.nn.functional.layer_norm(
            x_33,
            (384,),
            l_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_33 = l_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_35 = x_34.permute(0, 3, 1, 2)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = (None)
        split_2 = torch.functional.split(x_36, [384, 384, 5], 1)
        x_36 = None
        q_2 = split_2[0]
        ctx_2 = split_2[1]
        gates_2 = split_2[2]
        split_2 = None
        input_17 = torch.conv2d(
            ctx_2,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_2 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_18 = torch._C._nn.gelu(input_17, approximate="none")
        input_17 = None
        getitem_19 = gates_2[(slice(None, None, None), slice(0, 1, None))]
        mul_16 = input_18 * getitem_19
        getitem_19 = None
        ctx_all_10 = 0 + mul_16
        mul_16 = None
        input_19 = torch.conv2d(
            input_18,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_18 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_20 = torch._C._nn.gelu(input_19, approximate="none")
        input_19 = None
        getitem_20 = gates_2[(slice(None, None, None), slice(1, 2, None))]
        mul_17 = input_20 * getitem_20
        getitem_20 = None
        ctx_all_11 = ctx_all_10 + mul_17
        ctx_all_10 = mul_17 = None
        input_21 = torch.conv2d(
            input_20,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        input_20 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_22 = torch._C._nn.gelu(input_21, approximate="none")
        input_21 = None
        getitem_21 = gates_2[(slice(None, None, None), slice(2, 3, None))]
        mul_18 = input_22 * getitem_21
        getitem_21 = None
        ctx_all_12 = ctx_all_11 + mul_18
        ctx_all_11 = mul_18 = None
        input_23 = torch.conv2d(
            input_22,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            384,
        )
        input_22 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_24 = torch._C._nn.gelu(input_23, approximate="none")
        input_23 = None
        getitem_22 = gates_2[(slice(None, None, None), slice(3, 4, None))]
        mul_19 = input_24 * getitem_22
        getitem_22 = None
        ctx_all_13 = ctx_all_12 + mul_19
        ctx_all_12 = mul_19 = None
        mean_2 = input_24.mean((2, 3), keepdim=True)
        input_24 = None
        ctx_global_2 = torch._C._nn.gelu(mean_2, approximate="none")
        mean_2 = None
        getitem_23 = gates_2[(slice(None, None, None), slice(4, None, None))]
        gates_2 = None
        mul_20 = ctx_global_2 * getitem_23
        ctx_global_2 = getitem_23 = None
        ctx_all_14 = ctx_all_13 + mul_20
        ctx_all_13 = mul_20 = None
        conv2d_25 = torch.conv2d(
            ctx_all_14,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_14 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_6 = q_2 * conv2d_25
        q_2 = conv2d_25 = None
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
        x_37 = x_out_8.permute(0, 2, 3, 1)
        x_out_8 = None
        x_38 = torch.nn.functional.layer_norm(
            x_37,
            (384,),
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_37 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_post_parameters_bias_ = (None)
        x_39 = x_38.permute(0, 3, 1, 2)
        x_38 = None
        gamma_4 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_22 = x_39 * gamma_4
        x_39 = gamma_4 = None
        x_40 = x_35 + mul_22
        x_35 = mul_22 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_42 = torch._C._nn.gelu(x_41, approximate="none")
        x_41 = None
        x_43 = torch.nn.functional.dropout(x_42, 0.0, False, False)
        x_42 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_45 = torch.nn.functional.dropout(x_44, 0.0, False, False)
        x_44 = None
        x_46 = x_45.permute(0, 2, 3, 1)
        x_45 = None
        x_47 = torch.nn.functional.layer_norm(
            x_46,
            (384,),
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_46 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_post_parameters_bias_ = (None)
        x_48 = x_47.permute(0, 3, 1, 2)
        x_47 = None
        gamma_5 = l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_1_modules_blocks_modules_0_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_23 = x_48 * gamma_5
        x_48 = gamma_5 = None
        x_49 = x_40 + mul_23
        x_40 = mul_23 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = (None)
        split_3 = torch.functional.split(x_50, [384, 384, 5], 1)
        x_50 = None
        q_3 = split_3[0]
        ctx_3 = split_3[1]
        gates_3 = split_3[2]
        split_3 = None
        input_25 = torch.conv2d(
            ctx_3,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        ctx_3 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_26 = torch._C._nn.gelu(input_25, approximate="none")
        input_25 = None
        getitem_27 = gates_3[(slice(None, None, None), slice(0, 1, None))]
        mul_24 = input_26 * getitem_27
        getitem_27 = None
        ctx_all_15 = 0 + mul_24
        mul_24 = None
        input_27 = torch.conv2d(
            input_26,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        input_26 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_28 = torch._C._nn.gelu(input_27, approximate="none")
        input_27 = None
        getitem_28 = gates_3[(slice(None, None, None), slice(1, 2, None))]
        mul_25 = input_28 * getitem_28
        getitem_28 = None
        ctx_all_16 = ctx_all_15 + mul_25
        ctx_all_15 = mul_25 = None
        input_29 = torch.conv2d(
            input_28,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        input_28 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_30 = torch._C._nn.gelu(input_29, approximate="none")
        input_29 = None
        getitem_29 = gates_3[(slice(None, None, None), slice(2, 3, None))]
        mul_26 = input_30 * getitem_29
        getitem_29 = None
        ctx_all_17 = ctx_all_16 + mul_26
        ctx_all_16 = mul_26 = None
        input_31 = torch.conv2d(
            input_30,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            384,
        )
        input_30 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_32 = torch._C._nn.gelu(input_31, approximate="none")
        input_31 = None
        getitem_30 = gates_3[(slice(None, None, None), slice(3, 4, None))]
        mul_27 = input_32 * getitem_30
        getitem_30 = None
        ctx_all_18 = ctx_all_17 + mul_27
        ctx_all_17 = mul_27 = None
        mean_3 = input_32.mean((2, 3), keepdim=True)
        input_32 = None
        ctx_global_3 = torch._C._nn.gelu(mean_3, approximate="none")
        mean_3 = None
        getitem_31 = gates_3[(slice(None, None, None), slice(4, None, None))]
        gates_3 = None
        mul_28 = ctx_global_3 * getitem_31
        ctx_global_3 = getitem_31 = None
        ctx_all_19 = ctx_all_18 + mul_28
        ctx_all_18 = mul_28 = None
        conv2d_34 = torch.conv2d(
            ctx_all_19,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_19 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_9 = q_3 * conv2d_34
        q_3 = conv2d_34 = None
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
        x_51 = x_out_11.permute(0, 2, 3, 1)
        x_out_11 = None
        x_52 = torch.nn.functional.layer_norm(
            x_51,
            (384,),
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_51 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_post_parameters_bias_ = (None)
        x_53 = x_52.permute(0, 3, 1, 2)
        x_52 = None
        gamma_6 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_30 = x_53 * gamma_6
        x_53 = gamma_6 = None
        x_54 = x_49 + mul_30
        x_49 = mul_30 = None
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_56 = torch._C._nn.gelu(x_55, approximate="none")
        x_55 = None
        x_57 = torch.nn.functional.dropout(x_56, 0.0, False, False)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_59 = torch.nn.functional.dropout(x_58, 0.0, False, False)
        x_58 = None
        x_60 = x_59.permute(0, 2, 3, 1)
        x_59 = None
        x_61 = torch.nn.functional.layer_norm(
            x_60,
            (384,),
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_60 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_post_parameters_bias_ = (None)
        x_62 = x_61.permute(0, 3, 1, 2)
        x_61 = None
        gamma_7 = l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_1_modules_blocks_modules_1_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_31 = x_62 * gamma_7
        x_62 = gamma_7 = None
        x_63 = x_54 + mul_31
        x_54 = mul_31 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_bias_ = (None)
        x_65 = x_64.permute(0, 2, 3, 1)
        x_64 = None
        x_66 = torch.nn.functional.layer_norm(
            x_65,
            (768,),
            l_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_65 = l_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_67 = x_66.permute(0, 3, 1, 2)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = (None)
        split_4 = torch.functional.split(x_68, [768, 768, 5], 1)
        x_68 = None
        q_4 = split_4[0]
        ctx_4 = split_4[1]
        gates_4 = split_4[2]
        split_4 = None
        input_33 = torch.conv2d(
            ctx_4,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_4 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_34 = torch._C._nn.gelu(input_33, approximate="none")
        input_33 = None
        getitem_35 = gates_4[(slice(None, None, None), slice(0, 1, None))]
        mul_32 = input_34 * getitem_35
        getitem_35 = None
        ctx_all_20 = 0 + mul_32
        mul_32 = None
        input_35 = torch.conv2d(
            input_34,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_34 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_36 = torch._C._nn.gelu(input_35, approximate="none")
        input_35 = None
        getitem_36 = gates_4[(slice(None, None, None), slice(1, 2, None))]
        mul_33 = input_36 * getitem_36
        getitem_36 = None
        ctx_all_21 = ctx_all_20 + mul_33
        ctx_all_20 = mul_33 = None
        input_37 = torch.conv2d(
            input_36,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_36 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_38 = torch._C._nn.gelu(input_37, approximate="none")
        input_37 = None
        getitem_37 = gates_4[(slice(None, None, None), slice(2, 3, None))]
        mul_34 = input_38 * getitem_37
        getitem_37 = None
        ctx_all_22 = ctx_all_21 + mul_34
        ctx_all_21 = mul_34 = None
        input_39 = torch.conv2d(
            input_38,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_38 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_40 = torch._C._nn.gelu(input_39, approximate="none")
        input_39 = None
        getitem_38 = gates_4[(slice(None, None, None), slice(3, 4, None))]
        mul_35 = input_40 * getitem_38
        getitem_38 = None
        ctx_all_23 = ctx_all_22 + mul_35
        ctx_all_22 = mul_35 = None
        mean_4 = input_40.mean((2, 3), keepdim=True)
        input_40 = None
        ctx_global_4 = torch._C._nn.gelu(mean_4, approximate="none")
        mean_4 = None
        getitem_39 = gates_4[(slice(None, None, None), slice(4, None, None))]
        gates_4 = None
        mul_36 = ctx_global_4 * getitem_39
        ctx_global_4 = getitem_39 = None
        ctx_all_24 = ctx_all_23 + mul_36
        ctx_all_23 = mul_36 = None
        conv2d_44 = torch.conv2d(
            ctx_all_24,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_24 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_12 = q_4 * conv2d_44
        q_4 = conv2d_44 = None
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
        x_69 = x_out_14.permute(0, 2, 3, 1)
        x_out_14 = None
        x_70 = torch.nn.functional.layer_norm(
            x_69,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_69 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_post_parameters_bias_ = (None)
        x_71 = x_70.permute(0, 3, 1, 2)
        x_70 = None
        gamma_8 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_38 = x_71 * gamma_8
        x_71 = gamma_8 = None
        x_72 = x_67 + mul_38
        x_67 = mul_38 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_74 = torch._C._nn.gelu(x_73, approximate="none")
        x_73 = None
        x_75 = torch.nn.functional.dropout(x_74, 0.0, False, False)
        x_74 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_77 = torch.nn.functional.dropout(x_76, 0.0, False, False)
        x_76 = None
        x_78 = x_77.permute(0, 2, 3, 1)
        x_77 = None
        x_79 = torch.nn.functional.layer_norm(
            x_78,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_78 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_post_parameters_bias_ = (None)
        x_80 = x_79.permute(0, 3, 1, 2)
        x_79 = None
        gamma_9 = l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_0_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_39 = x_80 * gamma_9
        x_80 = gamma_9 = None
        x_81 = x_72 + mul_39
        x_72 = mul_39 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = (None)
        split_5 = torch.functional.split(x_82, [768, 768, 5], 1)
        x_82 = None
        q_5 = split_5[0]
        ctx_5 = split_5[1]
        gates_5 = split_5[2]
        split_5 = None
        input_41 = torch.conv2d(
            ctx_5,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_5 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_42 = torch._C._nn.gelu(input_41, approximate="none")
        input_41 = None
        getitem_43 = gates_5[(slice(None, None, None), slice(0, 1, None))]
        mul_40 = input_42 * getitem_43
        getitem_43 = None
        ctx_all_25 = 0 + mul_40
        mul_40 = None
        input_43 = torch.conv2d(
            input_42,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_42 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_44 = torch._C._nn.gelu(input_43, approximate="none")
        input_43 = None
        getitem_44 = gates_5[(slice(None, None, None), slice(1, 2, None))]
        mul_41 = input_44 * getitem_44
        getitem_44 = None
        ctx_all_26 = ctx_all_25 + mul_41
        ctx_all_25 = mul_41 = None
        input_45 = torch.conv2d(
            input_44,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_44 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_46 = torch._C._nn.gelu(input_45, approximate="none")
        input_45 = None
        getitem_45 = gates_5[(slice(None, None, None), slice(2, 3, None))]
        mul_42 = input_46 * getitem_45
        getitem_45 = None
        ctx_all_27 = ctx_all_26 + mul_42
        ctx_all_26 = mul_42 = None
        input_47 = torch.conv2d(
            input_46,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_46 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_48 = torch._C._nn.gelu(input_47, approximate="none")
        input_47 = None
        getitem_46 = gates_5[(slice(None, None, None), slice(3, 4, None))]
        mul_43 = input_48 * getitem_46
        getitem_46 = None
        ctx_all_28 = ctx_all_27 + mul_43
        ctx_all_27 = mul_43 = None
        mean_5 = input_48.mean((2, 3), keepdim=True)
        input_48 = None
        ctx_global_5 = torch._C._nn.gelu(mean_5, approximate="none")
        mean_5 = None
        getitem_47 = gates_5[(slice(None, None, None), slice(4, None, None))]
        gates_5 = None
        mul_44 = ctx_global_5 * getitem_47
        ctx_global_5 = getitem_47 = None
        ctx_all_29 = ctx_all_28 + mul_44
        ctx_all_28 = mul_44 = None
        conv2d_53 = torch.conv2d(
            ctx_all_29,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_29 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_15 = q_5 * conv2d_53
        q_5 = conv2d_53 = None
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
        x_83 = x_out_17.permute(0, 2, 3, 1)
        x_out_17 = None
        x_84 = torch.nn.functional.layer_norm(
            x_83,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_83 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_post_parameters_bias_ = (None)
        x_85 = x_84.permute(0, 3, 1, 2)
        x_84 = None
        gamma_10 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_46 = x_85 * gamma_10
        x_85 = gamma_10 = None
        x_86 = x_81 + mul_46
        x_81 = mul_46 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_88 = torch._C._nn.gelu(x_87, approximate="none")
        x_87 = None
        x_89 = torch.nn.functional.dropout(x_88, 0.0, False, False)
        x_88 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        x_92 = x_91.permute(0, 2, 3, 1)
        x_91 = None
        x_93 = torch.nn.functional.layer_norm(
            x_92,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_92 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_post_parameters_bias_ = (None)
        x_94 = x_93.permute(0, 3, 1, 2)
        x_93 = None
        gamma_11 = l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_1_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_47 = x_94 * gamma_11
        x_94 = gamma_11 = None
        x_95 = x_86 + mul_47
        x_86 = mul_47 = None
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_bias_ = (None)
        split_6 = torch.functional.split(x_96, [768, 768, 5], 1)
        x_96 = None
        q_6 = split_6[0]
        ctx_6 = split_6[1]
        gates_6 = split_6[2]
        split_6 = None
        input_49 = torch.conv2d(
            ctx_6,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_6 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_50 = torch._C._nn.gelu(input_49, approximate="none")
        input_49 = None
        getitem_51 = gates_6[(slice(None, None, None), slice(0, 1, None))]
        mul_48 = input_50 * getitem_51
        getitem_51 = None
        ctx_all_30 = 0 + mul_48
        mul_48 = None
        input_51 = torch.conv2d(
            input_50,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_50 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_52 = torch._C._nn.gelu(input_51, approximate="none")
        input_51 = None
        getitem_52 = gates_6[(slice(None, None, None), slice(1, 2, None))]
        mul_49 = input_52 * getitem_52
        getitem_52 = None
        ctx_all_31 = ctx_all_30 + mul_49
        ctx_all_30 = mul_49 = None
        input_53 = torch.conv2d(
            input_52,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_52 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_54 = torch._C._nn.gelu(input_53, approximate="none")
        input_53 = None
        getitem_53 = gates_6[(slice(None, None, None), slice(2, 3, None))]
        mul_50 = input_54 * getitem_53
        getitem_53 = None
        ctx_all_32 = ctx_all_31 + mul_50
        ctx_all_31 = mul_50 = None
        input_55 = torch.conv2d(
            input_54,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_54 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_56 = torch._C._nn.gelu(input_55, approximate="none")
        input_55 = None
        getitem_54 = gates_6[(slice(None, None, None), slice(3, 4, None))]
        mul_51 = input_56 * getitem_54
        getitem_54 = None
        ctx_all_33 = ctx_all_32 + mul_51
        ctx_all_32 = mul_51 = None
        mean_6 = input_56.mean((2, 3), keepdim=True)
        input_56 = None
        ctx_global_6 = torch._C._nn.gelu(mean_6, approximate="none")
        mean_6 = None
        getitem_55 = gates_6[(slice(None, None, None), slice(4, None, None))]
        gates_6 = None
        mul_52 = ctx_global_6 * getitem_55
        ctx_global_6 = getitem_55 = None
        ctx_all_34 = ctx_all_33 + mul_52
        ctx_all_33 = mul_52 = None
        conv2d_62 = torch.conv2d(
            ctx_all_34,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_34 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_18 = q_6 * conv2d_62
        q_6 = conv2d_62 = None
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
        x_97 = x_out_20.permute(0, 2, 3, 1)
        x_out_20 = None
        x_98 = torch.nn.functional.layer_norm(
            x_97,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_97 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_post_parameters_bias_ = (None)
        x_99 = x_98.permute(0, 3, 1, 2)
        x_98 = None
        gamma_12 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_54 = x_99 * gamma_12
        x_99 = gamma_12 = None
        x_100 = x_95 + mul_54
        x_95 = mul_54 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_102 = torch._C._nn.gelu(x_101, approximate="none")
        x_101 = None
        x_103 = torch.nn.functional.dropout(x_102, 0.0, False, False)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_105 = torch.nn.functional.dropout(x_104, 0.0, False, False)
        x_104 = None
        x_106 = x_105.permute(0, 2, 3, 1)
        x_105 = None
        x_107 = torch.nn.functional.layer_norm(
            x_106,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_106 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_post_parameters_bias_ = (None)
        x_108 = x_107.permute(0, 3, 1, 2)
        x_107 = None
        gamma_13 = l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_2_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_55 = x_108 * gamma_13
        x_108 = gamma_13 = None
        x_109 = x_100 + mul_55
        x_100 = mul_55 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_bias_ = (None)
        split_7 = torch.functional.split(x_110, [768, 768, 5], 1)
        x_110 = None
        q_7 = split_7[0]
        ctx_7 = split_7[1]
        gates_7 = split_7[2]
        split_7 = None
        input_57 = torch.conv2d(
            ctx_7,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_7 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_58 = torch._C._nn.gelu(input_57, approximate="none")
        input_57 = None
        getitem_59 = gates_7[(slice(None, None, None), slice(0, 1, None))]
        mul_56 = input_58 * getitem_59
        getitem_59 = None
        ctx_all_35 = 0 + mul_56
        mul_56 = None
        input_59 = torch.conv2d(
            input_58,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_58 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_60 = torch._C._nn.gelu(input_59, approximate="none")
        input_59 = None
        getitem_60 = gates_7[(slice(None, None, None), slice(1, 2, None))]
        mul_57 = input_60 * getitem_60
        getitem_60 = None
        ctx_all_36 = ctx_all_35 + mul_57
        ctx_all_35 = mul_57 = None
        input_61 = torch.conv2d(
            input_60,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_60 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_62 = torch._C._nn.gelu(input_61, approximate="none")
        input_61 = None
        getitem_61 = gates_7[(slice(None, None, None), slice(2, 3, None))]
        mul_58 = input_62 * getitem_61
        getitem_61 = None
        ctx_all_37 = ctx_all_36 + mul_58
        ctx_all_36 = mul_58 = None
        input_63 = torch.conv2d(
            input_62,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_62 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_64 = torch._C._nn.gelu(input_63, approximate="none")
        input_63 = None
        getitem_62 = gates_7[(slice(None, None, None), slice(3, 4, None))]
        mul_59 = input_64 * getitem_62
        getitem_62 = None
        ctx_all_38 = ctx_all_37 + mul_59
        ctx_all_37 = mul_59 = None
        mean_7 = input_64.mean((2, 3), keepdim=True)
        input_64 = None
        ctx_global_7 = torch._C._nn.gelu(mean_7, approximate="none")
        mean_7 = None
        getitem_63 = gates_7[(slice(None, None, None), slice(4, None, None))]
        gates_7 = None
        mul_60 = ctx_global_7 * getitem_63
        ctx_global_7 = getitem_63 = None
        ctx_all_39 = ctx_all_38 + mul_60
        ctx_all_38 = mul_60 = None
        conv2d_71 = torch.conv2d(
            ctx_all_39,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_39 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_21 = q_7 * conv2d_71
        q_7 = conv2d_71 = None
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
        x_111 = x_out_23.permute(0, 2, 3, 1)
        x_out_23 = None
        x_112 = torch.nn.functional.layer_norm(
            x_111,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_111 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_post_parameters_bias_ = (None)
        x_113 = x_112.permute(0, 3, 1, 2)
        x_112 = None
        gamma_14 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_62 = x_113 * gamma_14
        x_113 = gamma_14 = None
        x_114 = x_109 + mul_62
        x_109 = mul_62 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_116 = torch._C._nn.gelu(x_115, approximate="none")
        x_115 = None
        x_117 = torch.nn.functional.dropout(x_116, 0.0, False, False)
        x_116 = None
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_119 = torch.nn.functional.dropout(x_118, 0.0, False, False)
        x_118 = None
        x_120 = x_119.permute(0, 2, 3, 1)
        x_119 = None
        x_121 = torch.nn.functional.layer_norm(
            x_120,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_120 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_post_parameters_bias_ = (None)
        x_122 = x_121.permute(0, 3, 1, 2)
        x_121 = None
        gamma_15 = l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_3_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_63 = x_122 * gamma_15
        x_122 = gamma_15 = None
        x_123 = x_114 + mul_63
        x_114 = mul_63 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_bias_ = (None)
        split_8 = torch.functional.split(x_124, [768, 768, 5], 1)
        x_124 = None
        q_8 = split_8[0]
        ctx_8 = split_8[1]
        gates_8 = split_8[2]
        split_8 = None
        input_65 = torch.conv2d(
            ctx_8,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_8 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_66 = torch._C._nn.gelu(input_65, approximate="none")
        input_65 = None
        getitem_67 = gates_8[(slice(None, None, None), slice(0, 1, None))]
        mul_64 = input_66 * getitem_67
        getitem_67 = None
        ctx_all_40 = 0 + mul_64
        mul_64 = None
        input_67 = torch.conv2d(
            input_66,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_66 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_68 = torch._C._nn.gelu(input_67, approximate="none")
        input_67 = None
        getitem_68 = gates_8[(slice(None, None, None), slice(1, 2, None))]
        mul_65 = input_68 * getitem_68
        getitem_68 = None
        ctx_all_41 = ctx_all_40 + mul_65
        ctx_all_40 = mul_65 = None
        input_69 = torch.conv2d(
            input_68,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_68 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_70 = torch._C._nn.gelu(input_69, approximate="none")
        input_69 = None
        getitem_69 = gates_8[(slice(None, None, None), slice(2, 3, None))]
        mul_66 = input_70 * getitem_69
        getitem_69 = None
        ctx_all_42 = ctx_all_41 + mul_66
        ctx_all_41 = mul_66 = None
        input_71 = torch.conv2d(
            input_70,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_70 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_72 = torch._C._nn.gelu(input_71, approximate="none")
        input_71 = None
        getitem_70 = gates_8[(slice(None, None, None), slice(3, 4, None))]
        mul_67 = input_72 * getitem_70
        getitem_70 = None
        ctx_all_43 = ctx_all_42 + mul_67
        ctx_all_42 = mul_67 = None
        mean_8 = input_72.mean((2, 3), keepdim=True)
        input_72 = None
        ctx_global_8 = torch._C._nn.gelu(mean_8, approximate="none")
        mean_8 = None
        getitem_71 = gates_8[(slice(None, None, None), slice(4, None, None))]
        gates_8 = None
        mul_68 = ctx_global_8 * getitem_71
        ctx_global_8 = getitem_71 = None
        ctx_all_44 = ctx_all_43 + mul_68
        ctx_all_43 = mul_68 = None
        conv2d_80 = torch.conv2d(
            ctx_all_44,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_44 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_24 = q_8 * conv2d_80
        q_8 = conv2d_80 = None
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
        x_125 = x_out_26.permute(0, 2, 3, 1)
        x_out_26 = None
        x_126 = torch.nn.functional.layer_norm(
            x_125,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_125 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_post_parameters_bias_ = (None)
        x_127 = x_126.permute(0, 3, 1, 2)
        x_126 = None
        gamma_16 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_70 = x_127 * gamma_16
        x_127 = gamma_16 = None
        x_128 = x_123 + mul_70
        x_123 = mul_70 = None
        x_129 = torch.conv2d(
            x_128,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_130 = torch._C._nn.gelu(x_129, approximate="none")
        x_129 = None
        x_131 = torch.nn.functional.dropout(x_130, 0.0, False, False)
        x_130 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_131 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_133 = torch.nn.functional.dropout(x_132, 0.0, False, False)
        x_132 = None
        x_134 = x_133.permute(0, 2, 3, 1)
        x_133 = None
        x_135 = torch.nn.functional.layer_norm(
            x_134,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_134 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_post_parameters_bias_ = (None)
        x_136 = x_135.permute(0, 3, 1, 2)
        x_135 = None
        gamma_17 = l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_4_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_71 = x_136 * gamma_17
        x_136 = gamma_17 = None
        x_137 = x_128 + mul_71
        x_128 = mul_71 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_bias_ = (None)
        split_9 = torch.functional.split(x_138, [768, 768, 5], 1)
        x_138 = None
        q_9 = split_9[0]
        ctx_9 = split_9[1]
        gates_9 = split_9[2]
        split_9 = None
        input_73 = torch.conv2d(
            ctx_9,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_9 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_74 = torch._C._nn.gelu(input_73, approximate="none")
        input_73 = None
        getitem_75 = gates_9[(slice(None, None, None), slice(0, 1, None))]
        mul_72 = input_74 * getitem_75
        getitem_75 = None
        ctx_all_45 = 0 + mul_72
        mul_72 = None
        input_75 = torch.conv2d(
            input_74,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_74 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_76 = torch._C._nn.gelu(input_75, approximate="none")
        input_75 = None
        getitem_76 = gates_9[(slice(None, None, None), slice(1, 2, None))]
        mul_73 = input_76 * getitem_76
        getitem_76 = None
        ctx_all_46 = ctx_all_45 + mul_73
        ctx_all_45 = mul_73 = None
        input_77 = torch.conv2d(
            input_76,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_76 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_78 = torch._C._nn.gelu(input_77, approximate="none")
        input_77 = None
        getitem_77 = gates_9[(slice(None, None, None), slice(2, 3, None))]
        mul_74 = input_78 * getitem_77
        getitem_77 = None
        ctx_all_47 = ctx_all_46 + mul_74
        ctx_all_46 = mul_74 = None
        input_79 = torch.conv2d(
            input_78,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_78 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_80 = torch._C._nn.gelu(input_79, approximate="none")
        input_79 = None
        getitem_78 = gates_9[(slice(None, None, None), slice(3, 4, None))]
        mul_75 = input_80 * getitem_78
        getitem_78 = None
        ctx_all_48 = ctx_all_47 + mul_75
        ctx_all_47 = mul_75 = None
        mean_9 = input_80.mean((2, 3), keepdim=True)
        input_80 = None
        ctx_global_9 = torch._C._nn.gelu(mean_9, approximate="none")
        mean_9 = None
        getitem_79 = gates_9[(slice(None, None, None), slice(4, None, None))]
        gates_9 = None
        mul_76 = ctx_global_9 * getitem_79
        ctx_global_9 = getitem_79 = None
        ctx_all_49 = ctx_all_48 + mul_76
        ctx_all_48 = mul_76 = None
        conv2d_89 = torch.conv2d(
            ctx_all_49,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_49 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_27 = q_9 * conv2d_89
        q_9 = conv2d_89 = None
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
        x_139 = x_out_29.permute(0, 2, 3, 1)
        x_out_29 = None
        x_140 = torch.nn.functional.layer_norm(
            x_139,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_139 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_post_parameters_bias_ = (None)
        x_141 = x_140.permute(0, 3, 1, 2)
        x_140 = None
        gamma_18 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_78 = x_141 * gamma_18
        x_141 = gamma_18 = None
        x_142 = x_137 + mul_78
        x_137 = mul_78 = None
        x_143 = torch.conv2d(
            x_142,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_144 = torch._C._nn.gelu(x_143, approximate="none")
        x_143 = None
        x_145 = torch.nn.functional.dropout(x_144, 0.0, False, False)
        x_144 = None
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_145 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_147 = torch.nn.functional.dropout(x_146, 0.0, False, False)
        x_146 = None
        x_148 = x_147.permute(0, 2, 3, 1)
        x_147 = None
        x_149 = torch.nn.functional.layer_norm(
            x_148,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_148 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_post_parameters_bias_ = (None)
        x_150 = x_149.permute(0, 3, 1, 2)
        x_149 = None
        gamma_19 = l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_5_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_79 = x_150 * gamma_19
        x_150 = gamma_19 = None
        x_151 = x_142 + mul_79
        x_142 = mul_79 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_bias_ = (None)
        split_10 = torch.functional.split(x_152, [768, 768, 5], 1)
        x_152 = None
        q_10 = split_10[0]
        ctx_10 = split_10[1]
        gates_10 = split_10[2]
        split_10 = None
        input_81 = torch.conv2d(
            ctx_10,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_10 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_82 = torch._C._nn.gelu(input_81, approximate="none")
        input_81 = None
        getitem_83 = gates_10[(slice(None, None, None), slice(0, 1, None))]
        mul_80 = input_82 * getitem_83
        getitem_83 = None
        ctx_all_50 = 0 + mul_80
        mul_80 = None
        input_83 = torch.conv2d(
            input_82,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_82 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_84 = torch._C._nn.gelu(input_83, approximate="none")
        input_83 = None
        getitem_84 = gates_10[(slice(None, None, None), slice(1, 2, None))]
        mul_81 = input_84 * getitem_84
        getitem_84 = None
        ctx_all_51 = ctx_all_50 + mul_81
        ctx_all_50 = mul_81 = None
        input_85 = torch.conv2d(
            input_84,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_84 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_86 = torch._C._nn.gelu(input_85, approximate="none")
        input_85 = None
        getitem_85 = gates_10[(slice(None, None, None), slice(2, 3, None))]
        mul_82 = input_86 * getitem_85
        getitem_85 = None
        ctx_all_52 = ctx_all_51 + mul_82
        ctx_all_51 = mul_82 = None
        input_87 = torch.conv2d(
            input_86,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_86 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_88 = torch._C._nn.gelu(input_87, approximate="none")
        input_87 = None
        getitem_86 = gates_10[(slice(None, None, None), slice(3, 4, None))]
        mul_83 = input_88 * getitem_86
        getitem_86 = None
        ctx_all_53 = ctx_all_52 + mul_83
        ctx_all_52 = mul_83 = None
        mean_10 = input_88.mean((2, 3), keepdim=True)
        input_88 = None
        ctx_global_10 = torch._C._nn.gelu(mean_10, approximate="none")
        mean_10 = None
        getitem_87 = gates_10[(slice(None, None, None), slice(4, None, None))]
        gates_10 = None
        mul_84 = ctx_global_10 * getitem_87
        ctx_global_10 = getitem_87 = None
        ctx_all_54 = ctx_all_53 + mul_84
        ctx_all_53 = mul_84 = None
        conv2d_98 = torch.conv2d(
            ctx_all_54,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_54 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_30 = q_10 * conv2d_98
        q_10 = conv2d_98 = None
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
        x_153 = x_out_32.permute(0, 2, 3, 1)
        x_out_32 = None
        x_154 = torch.nn.functional.layer_norm(
            x_153,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_153 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_post_parameters_bias_ = (None)
        x_155 = x_154.permute(0, 3, 1, 2)
        x_154 = None
        gamma_20 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_86 = x_155 * gamma_20
        x_155 = gamma_20 = None
        x_156 = x_151 + mul_86
        x_151 = mul_86 = None
        x_157 = torch.conv2d(
            x_156,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_158 = torch._C._nn.gelu(x_157, approximate="none")
        x_157 = None
        x_159 = torch.nn.functional.dropout(x_158, 0.0, False, False)
        x_158 = None
        x_160 = torch.conv2d(
            x_159,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_159 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_161 = torch.nn.functional.dropout(x_160, 0.0, False, False)
        x_160 = None
        x_162 = x_161.permute(0, 2, 3, 1)
        x_161 = None
        x_163 = torch.nn.functional.layer_norm(
            x_162,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_162 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_post_parameters_bias_ = (None)
        x_164 = x_163.permute(0, 3, 1, 2)
        x_163 = None
        gamma_21 = l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_6_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_87 = x_164 * gamma_21
        x_164 = gamma_21 = None
        x_165 = x_156 + mul_87
        x_156 = mul_87 = None
        x_166 = torch.conv2d(
            x_165,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_bias_ = (None)
        split_11 = torch.functional.split(x_166, [768, 768, 5], 1)
        x_166 = None
        q_11 = split_11[0]
        ctx_11 = split_11[1]
        gates_11 = split_11[2]
        split_11 = None
        input_89 = torch.conv2d(
            ctx_11,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_11 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_90 = torch._C._nn.gelu(input_89, approximate="none")
        input_89 = None
        getitem_91 = gates_11[(slice(None, None, None), slice(0, 1, None))]
        mul_88 = input_90 * getitem_91
        getitem_91 = None
        ctx_all_55 = 0 + mul_88
        mul_88 = None
        input_91 = torch.conv2d(
            input_90,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_90 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_92 = torch._C._nn.gelu(input_91, approximate="none")
        input_91 = None
        getitem_92 = gates_11[(slice(None, None, None), slice(1, 2, None))]
        mul_89 = input_92 * getitem_92
        getitem_92 = None
        ctx_all_56 = ctx_all_55 + mul_89
        ctx_all_55 = mul_89 = None
        input_93 = torch.conv2d(
            input_92,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_92 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_94 = torch._C._nn.gelu(input_93, approximate="none")
        input_93 = None
        getitem_93 = gates_11[(slice(None, None, None), slice(2, 3, None))]
        mul_90 = input_94 * getitem_93
        getitem_93 = None
        ctx_all_57 = ctx_all_56 + mul_90
        ctx_all_56 = mul_90 = None
        input_95 = torch.conv2d(
            input_94,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_94 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_96 = torch._C._nn.gelu(input_95, approximate="none")
        input_95 = None
        getitem_94 = gates_11[(slice(None, None, None), slice(3, 4, None))]
        mul_91 = input_96 * getitem_94
        getitem_94 = None
        ctx_all_58 = ctx_all_57 + mul_91
        ctx_all_57 = mul_91 = None
        mean_11 = input_96.mean((2, 3), keepdim=True)
        input_96 = None
        ctx_global_11 = torch._C._nn.gelu(mean_11, approximate="none")
        mean_11 = None
        getitem_95 = gates_11[(slice(None, None, None), slice(4, None, None))]
        gates_11 = None
        mul_92 = ctx_global_11 * getitem_95
        ctx_global_11 = getitem_95 = None
        ctx_all_59 = ctx_all_58 + mul_92
        ctx_all_58 = mul_92 = None
        conv2d_107 = torch.conv2d(
            ctx_all_59,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_59 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_33 = q_11 * conv2d_107
        q_11 = conv2d_107 = None
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
        x_167 = x_out_35.permute(0, 2, 3, 1)
        x_out_35 = None
        x_168 = torch.nn.functional.layer_norm(
            x_167,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_167 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_post_parameters_bias_ = (None)
        x_169 = x_168.permute(0, 3, 1, 2)
        x_168 = None
        gamma_22 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_94 = x_169 * gamma_22
        x_169 = gamma_22 = None
        x_170 = x_165 + mul_94
        x_165 = mul_94 = None
        x_171 = torch.conv2d(
            x_170,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_172 = torch._C._nn.gelu(x_171, approximate="none")
        x_171 = None
        x_173 = torch.nn.functional.dropout(x_172, 0.0, False, False)
        x_172 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_173 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_175 = torch.nn.functional.dropout(x_174, 0.0, False, False)
        x_174 = None
        x_176 = x_175.permute(0, 2, 3, 1)
        x_175 = None
        x_177 = torch.nn.functional.layer_norm(
            x_176,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_176 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_post_parameters_bias_ = (None)
        x_178 = x_177.permute(0, 3, 1, 2)
        x_177 = None
        gamma_23 = l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_7_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_95 = x_178 * gamma_23
        x_178 = gamma_23 = None
        x_179 = x_170 + mul_95
        x_170 = mul_95 = None
        x_180 = torch.conv2d(
            x_179,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_bias_ = (None)
        split_12 = torch.functional.split(x_180, [768, 768, 5], 1)
        x_180 = None
        q_12 = split_12[0]
        ctx_12 = split_12[1]
        gates_12 = split_12[2]
        split_12 = None
        input_97 = torch.conv2d(
            ctx_12,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_12 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_98 = torch._C._nn.gelu(input_97, approximate="none")
        input_97 = None
        getitem_99 = gates_12[(slice(None, None, None), slice(0, 1, None))]
        mul_96 = input_98 * getitem_99
        getitem_99 = None
        ctx_all_60 = 0 + mul_96
        mul_96 = None
        input_99 = torch.conv2d(
            input_98,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_98 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_100 = torch._C._nn.gelu(input_99, approximate="none")
        input_99 = None
        getitem_100 = gates_12[(slice(None, None, None), slice(1, 2, None))]
        mul_97 = input_100 * getitem_100
        getitem_100 = None
        ctx_all_61 = ctx_all_60 + mul_97
        ctx_all_60 = mul_97 = None
        input_101 = torch.conv2d(
            input_100,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_100 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_102 = torch._C._nn.gelu(input_101, approximate="none")
        input_101 = None
        getitem_101 = gates_12[(slice(None, None, None), slice(2, 3, None))]
        mul_98 = input_102 * getitem_101
        getitem_101 = None
        ctx_all_62 = ctx_all_61 + mul_98
        ctx_all_61 = mul_98 = None
        input_103 = torch.conv2d(
            input_102,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_102 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_104 = torch._C._nn.gelu(input_103, approximate="none")
        input_103 = None
        getitem_102 = gates_12[(slice(None, None, None), slice(3, 4, None))]
        mul_99 = input_104 * getitem_102
        getitem_102 = None
        ctx_all_63 = ctx_all_62 + mul_99
        ctx_all_62 = mul_99 = None
        mean_12 = input_104.mean((2, 3), keepdim=True)
        input_104 = None
        ctx_global_12 = torch._C._nn.gelu(mean_12, approximate="none")
        mean_12 = None
        getitem_103 = gates_12[(slice(None, None, None), slice(4, None, None))]
        gates_12 = None
        mul_100 = ctx_global_12 * getitem_103
        ctx_global_12 = getitem_103 = None
        ctx_all_64 = ctx_all_63 + mul_100
        ctx_all_63 = mul_100 = None
        conv2d_116 = torch.conv2d(
            ctx_all_64,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_64 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_36 = q_12 * conv2d_116
        q_12 = conv2d_116 = None
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
        x_181 = x_out_38.permute(0, 2, 3, 1)
        x_out_38 = None
        x_182 = torch.nn.functional.layer_norm(
            x_181,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_181 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_post_parameters_bias_ = (None)
        x_183 = x_182.permute(0, 3, 1, 2)
        x_182 = None
        gamma_24 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_102 = x_183 * gamma_24
        x_183 = gamma_24 = None
        x_184 = x_179 + mul_102
        x_179 = mul_102 = None
        x_185 = torch.conv2d(
            x_184,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_186 = torch._C._nn.gelu(x_185, approximate="none")
        x_185 = None
        x_187 = torch.nn.functional.dropout(x_186, 0.0, False, False)
        x_186 = None
        x_188 = torch.conv2d(
            x_187,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_187 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_189 = torch.nn.functional.dropout(x_188, 0.0, False, False)
        x_188 = None
        x_190 = x_189.permute(0, 2, 3, 1)
        x_189 = None
        x_191 = torch.nn.functional.layer_norm(
            x_190,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_190 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_post_parameters_bias_ = (None)
        x_192 = x_191.permute(0, 3, 1, 2)
        x_191 = None
        gamma_25 = l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_8_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_103 = x_192 * gamma_25
        x_192 = gamma_25 = None
        x_193 = x_184 + mul_103
        x_184 = mul_103 = None
        x_194 = torch.conv2d(
            x_193,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_bias_ = (None)
        split_13 = torch.functional.split(x_194, [768, 768, 5], 1)
        x_194 = None
        q_13 = split_13[0]
        ctx_13 = split_13[1]
        gates_13 = split_13[2]
        split_13 = None
        input_105 = torch.conv2d(
            ctx_13,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_13 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_106 = torch._C._nn.gelu(input_105, approximate="none")
        input_105 = None
        getitem_107 = gates_13[(slice(None, None, None), slice(0, 1, None))]
        mul_104 = input_106 * getitem_107
        getitem_107 = None
        ctx_all_65 = 0 + mul_104
        mul_104 = None
        input_107 = torch.conv2d(
            input_106,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_106 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_108 = torch._C._nn.gelu(input_107, approximate="none")
        input_107 = None
        getitem_108 = gates_13[(slice(None, None, None), slice(1, 2, None))]
        mul_105 = input_108 * getitem_108
        getitem_108 = None
        ctx_all_66 = ctx_all_65 + mul_105
        ctx_all_65 = mul_105 = None
        input_109 = torch.conv2d(
            input_108,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_108 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_110 = torch._C._nn.gelu(input_109, approximate="none")
        input_109 = None
        getitem_109 = gates_13[(slice(None, None, None), slice(2, 3, None))]
        mul_106 = input_110 * getitem_109
        getitem_109 = None
        ctx_all_67 = ctx_all_66 + mul_106
        ctx_all_66 = mul_106 = None
        input_111 = torch.conv2d(
            input_110,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_110 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_112 = torch._C._nn.gelu(input_111, approximate="none")
        input_111 = None
        getitem_110 = gates_13[(slice(None, None, None), slice(3, 4, None))]
        mul_107 = input_112 * getitem_110
        getitem_110 = None
        ctx_all_68 = ctx_all_67 + mul_107
        ctx_all_67 = mul_107 = None
        mean_13 = input_112.mean((2, 3), keepdim=True)
        input_112 = None
        ctx_global_13 = torch._C._nn.gelu(mean_13, approximate="none")
        mean_13 = None
        getitem_111 = gates_13[(slice(None, None, None), slice(4, None, None))]
        gates_13 = None
        mul_108 = ctx_global_13 * getitem_111
        ctx_global_13 = getitem_111 = None
        ctx_all_69 = ctx_all_68 + mul_108
        ctx_all_68 = mul_108 = None
        conv2d_125 = torch.conv2d(
            ctx_all_69,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_69 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_39 = q_13 * conv2d_125
        q_13 = conv2d_125 = None
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
        x_195 = x_out_41.permute(0, 2, 3, 1)
        x_out_41 = None
        x_196 = torch.nn.functional.layer_norm(
            x_195,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_195 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_post_parameters_bias_ = (None)
        x_197 = x_196.permute(0, 3, 1, 2)
        x_196 = None
        gamma_26 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_110 = x_197 * gamma_26
        x_197 = gamma_26 = None
        x_198 = x_193 + mul_110
        x_193 = mul_110 = None
        x_199 = torch.conv2d(
            x_198,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_200 = torch._C._nn.gelu(x_199, approximate="none")
        x_199 = None
        x_201 = torch.nn.functional.dropout(x_200, 0.0, False, False)
        x_200 = None
        x_202 = torch.conv2d(
            x_201,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_201 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_203 = torch.nn.functional.dropout(x_202, 0.0, False, False)
        x_202 = None
        x_204 = x_203.permute(0, 2, 3, 1)
        x_203 = None
        x_205 = torch.nn.functional.layer_norm(
            x_204,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_204 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_post_parameters_bias_ = (None)
        x_206 = x_205.permute(0, 3, 1, 2)
        x_205 = None
        gamma_27 = l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_9_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_111 = x_206 * gamma_27
        x_206 = gamma_27 = None
        x_207 = x_198 + mul_111
        x_198 = mul_111 = None
        x_208 = torch.conv2d(
            x_207,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_bias_ = (None)
        split_14 = torch.functional.split(x_208, [768, 768, 5], 1)
        x_208 = None
        q_14 = split_14[0]
        ctx_14 = split_14[1]
        gates_14 = split_14[2]
        split_14 = None
        input_113 = torch.conv2d(
            ctx_14,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_14 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_114 = torch._C._nn.gelu(input_113, approximate="none")
        input_113 = None
        getitem_115 = gates_14[(slice(None, None, None), slice(0, 1, None))]
        mul_112 = input_114 * getitem_115
        getitem_115 = None
        ctx_all_70 = 0 + mul_112
        mul_112 = None
        input_115 = torch.conv2d(
            input_114,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_114 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_116 = torch._C._nn.gelu(input_115, approximate="none")
        input_115 = None
        getitem_116 = gates_14[(slice(None, None, None), slice(1, 2, None))]
        mul_113 = input_116 * getitem_116
        getitem_116 = None
        ctx_all_71 = ctx_all_70 + mul_113
        ctx_all_70 = mul_113 = None
        input_117 = torch.conv2d(
            input_116,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_116 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_118 = torch._C._nn.gelu(input_117, approximate="none")
        input_117 = None
        getitem_117 = gates_14[(slice(None, None, None), slice(2, 3, None))]
        mul_114 = input_118 * getitem_117
        getitem_117 = None
        ctx_all_72 = ctx_all_71 + mul_114
        ctx_all_71 = mul_114 = None
        input_119 = torch.conv2d(
            input_118,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_118 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_120 = torch._C._nn.gelu(input_119, approximate="none")
        input_119 = None
        getitem_118 = gates_14[(slice(None, None, None), slice(3, 4, None))]
        mul_115 = input_120 * getitem_118
        getitem_118 = None
        ctx_all_73 = ctx_all_72 + mul_115
        ctx_all_72 = mul_115 = None
        mean_14 = input_120.mean((2, 3), keepdim=True)
        input_120 = None
        ctx_global_14 = torch._C._nn.gelu(mean_14, approximate="none")
        mean_14 = None
        getitem_119 = gates_14[(slice(None, None, None), slice(4, None, None))]
        gates_14 = None
        mul_116 = ctx_global_14 * getitem_119
        ctx_global_14 = getitem_119 = None
        ctx_all_74 = ctx_all_73 + mul_116
        ctx_all_73 = mul_116 = None
        conv2d_134 = torch.conv2d(
            ctx_all_74,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_74 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_42 = q_14 * conv2d_134
        q_14 = conv2d_134 = None
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
        x_209 = x_out_44.permute(0, 2, 3, 1)
        x_out_44 = None
        x_210 = torch.nn.functional.layer_norm(
            x_209,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_209 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_post_parameters_bias_ = (None)
        x_211 = x_210.permute(0, 3, 1, 2)
        x_210 = None
        gamma_28 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_118 = x_211 * gamma_28
        x_211 = gamma_28 = None
        x_212 = x_207 + mul_118
        x_207 = mul_118 = None
        x_213 = torch.conv2d(
            x_212,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_214 = torch._C._nn.gelu(x_213, approximate="none")
        x_213 = None
        x_215 = torch.nn.functional.dropout(x_214, 0.0, False, False)
        x_214 = None
        x_216 = torch.conv2d(
            x_215,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_215 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_217 = torch.nn.functional.dropout(x_216, 0.0, False, False)
        x_216 = None
        x_218 = x_217.permute(0, 2, 3, 1)
        x_217 = None
        x_219 = torch.nn.functional.layer_norm(
            x_218,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_218 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_post_parameters_bias_ = (None)
        x_220 = x_219.permute(0, 3, 1, 2)
        x_219 = None
        gamma_29 = l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_10_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_119 = x_220 * gamma_29
        x_220 = gamma_29 = None
        x_221 = x_212 + mul_119
        x_212 = mul_119 = None
        x_222 = torch.conv2d(
            x_221,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_bias_ = (None)
        split_15 = torch.functional.split(x_222, [768, 768, 5], 1)
        x_222 = None
        q_15 = split_15[0]
        ctx_15 = split_15[1]
        gates_15 = split_15[2]
        split_15 = None
        input_121 = torch.conv2d(
            ctx_15,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_15 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_122 = torch._C._nn.gelu(input_121, approximate="none")
        input_121 = None
        getitem_123 = gates_15[(slice(None, None, None), slice(0, 1, None))]
        mul_120 = input_122 * getitem_123
        getitem_123 = None
        ctx_all_75 = 0 + mul_120
        mul_120 = None
        input_123 = torch.conv2d(
            input_122,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_122 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_124 = torch._C._nn.gelu(input_123, approximate="none")
        input_123 = None
        getitem_124 = gates_15[(slice(None, None, None), slice(1, 2, None))]
        mul_121 = input_124 * getitem_124
        getitem_124 = None
        ctx_all_76 = ctx_all_75 + mul_121
        ctx_all_75 = mul_121 = None
        input_125 = torch.conv2d(
            input_124,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_124 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_126 = torch._C._nn.gelu(input_125, approximate="none")
        input_125 = None
        getitem_125 = gates_15[(slice(None, None, None), slice(2, 3, None))]
        mul_122 = input_126 * getitem_125
        getitem_125 = None
        ctx_all_77 = ctx_all_76 + mul_122
        ctx_all_76 = mul_122 = None
        input_127 = torch.conv2d(
            input_126,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_126 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_128 = torch._C._nn.gelu(input_127, approximate="none")
        input_127 = None
        getitem_126 = gates_15[(slice(None, None, None), slice(3, 4, None))]
        mul_123 = input_128 * getitem_126
        getitem_126 = None
        ctx_all_78 = ctx_all_77 + mul_123
        ctx_all_77 = mul_123 = None
        mean_15 = input_128.mean((2, 3), keepdim=True)
        input_128 = None
        ctx_global_15 = torch._C._nn.gelu(mean_15, approximate="none")
        mean_15 = None
        getitem_127 = gates_15[(slice(None, None, None), slice(4, None, None))]
        gates_15 = None
        mul_124 = ctx_global_15 * getitem_127
        ctx_global_15 = getitem_127 = None
        ctx_all_79 = ctx_all_78 + mul_124
        ctx_all_78 = mul_124 = None
        conv2d_143 = torch.conv2d(
            ctx_all_79,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_79 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_45 = q_15 * conv2d_143
        q_15 = conv2d_143 = None
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
        x_223 = x_out_47.permute(0, 2, 3, 1)
        x_out_47 = None
        x_224 = torch.nn.functional.layer_norm(
            x_223,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_223 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_post_parameters_bias_ = (None)
        x_225 = x_224.permute(0, 3, 1, 2)
        x_224 = None
        gamma_30 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_126 = x_225 * gamma_30
        x_225 = gamma_30 = None
        x_226 = x_221 + mul_126
        x_221 = mul_126 = None
        x_227 = torch.conv2d(
            x_226,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_228 = torch._C._nn.gelu(x_227, approximate="none")
        x_227 = None
        x_229 = torch.nn.functional.dropout(x_228, 0.0, False, False)
        x_228 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_231 = torch.nn.functional.dropout(x_230, 0.0, False, False)
        x_230 = None
        x_232 = x_231.permute(0, 2, 3, 1)
        x_231 = None
        x_233 = torch.nn.functional.layer_norm(
            x_232,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_232 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_post_parameters_bias_ = (None)
        x_234 = x_233.permute(0, 3, 1, 2)
        x_233 = None
        gamma_31 = l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_11_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_127 = x_234 * gamma_31
        x_234 = gamma_31 = None
        x_235 = x_226 + mul_127
        x_226 = mul_127 = None
        x_236 = torch.conv2d(
            x_235,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_bias_ = (None)
        split_16 = torch.functional.split(x_236, [768, 768, 5], 1)
        x_236 = None
        q_16 = split_16[0]
        ctx_16 = split_16[1]
        gates_16 = split_16[2]
        split_16 = None
        input_129 = torch.conv2d(
            ctx_16,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_16 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_130 = torch._C._nn.gelu(input_129, approximate="none")
        input_129 = None
        getitem_131 = gates_16[(slice(None, None, None), slice(0, 1, None))]
        mul_128 = input_130 * getitem_131
        getitem_131 = None
        ctx_all_80 = 0 + mul_128
        mul_128 = None
        input_131 = torch.conv2d(
            input_130,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_130 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_132 = torch._C._nn.gelu(input_131, approximate="none")
        input_131 = None
        getitem_132 = gates_16[(slice(None, None, None), slice(1, 2, None))]
        mul_129 = input_132 * getitem_132
        getitem_132 = None
        ctx_all_81 = ctx_all_80 + mul_129
        ctx_all_80 = mul_129 = None
        input_133 = torch.conv2d(
            input_132,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_132 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_134 = torch._C._nn.gelu(input_133, approximate="none")
        input_133 = None
        getitem_133 = gates_16[(slice(None, None, None), slice(2, 3, None))]
        mul_130 = input_134 * getitem_133
        getitem_133 = None
        ctx_all_82 = ctx_all_81 + mul_130
        ctx_all_81 = mul_130 = None
        input_135 = torch.conv2d(
            input_134,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_134 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_136 = torch._C._nn.gelu(input_135, approximate="none")
        input_135 = None
        getitem_134 = gates_16[(slice(None, None, None), slice(3, 4, None))]
        mul_131 = input_136 * getitem_134
        getitem_134 = None
        ctx_all_83 = ctx_all_82 + mul_131
        ctx_all_82 = mul_131 = None
        mean_16 = input_136.mean((2, 3), keepdim=True)
        input_136 = None
        ctx_global_16 = torch._C._nn.gelu(mean_16, approximate="none")
        mean_16 = None
        getitem_135 = gates_16[(slice(None, None, None), slice(4, None, None))]
        gates_16 = None
        mul_132 = ctx_global_16 * getitem_135
        ctx_global_16 = getitem_135 = None
        ctx_all_84 = ctx_all_83 + mul_132
        ctx_all_83 = mul_132 = None
        conv2d_152 = torch.conv2d(
            ctx_all_84,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_84 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_48 = q_16 * conv2d_152
        q_16 = conv2d_152 = None
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
        x_237 = x_out_50.permute(0, 2, 3, 1)
        x_out_50 = None
        x_238 = torch.nn.functional.layer_norm(
            x_237,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_237 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_post_parameters_bias_ = (None)
        x_239 = x_238.permute(0, 3, 1, 2)
        x_238 = None
        gamma_32 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_134 = x_239 * gamma_32
        x_239 = gamma_32 = None
        x_240 = x_235 + mul_134
        x_235 = mul_134 = None
        x_241 = torch.conv2d(
            x_240,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_242 = torch._C._nn.gelu(x_241, approximate="none")
        x_241 = None
        x_243 = torch.nn.functional.dropout(x_242, 0.0, False, False)
        x_242 = None
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_243 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_245 = torch.nn.functional.dropout(x_244, 0.0, False, False)
        x_244 = None
        x_246 = x_245.permute(0, 2, 3, 1)
        x_245 = None
        x_247 = torch.nn.functional.layer_norm(
            x_246,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_246 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_post_parameters_bias_ = (None)
        x_248 = x_247.permute(0, 3, 1, 2)
        x_247 = None
        gamma_33 = l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_12_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_135 = x_248 * gamma_33
        x_248 = gamma_33 = None
        x_249 = x_240 + mul_135
        x_240 = mul_135 = None
        x_250 = torch.conv2d(
            x_249,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_bias_ = (None)
        split_17 = torch.functional.split(x_250, [768, 768, 5], 1)
        x_250 = None
        q_17 = split_17[0]
        ctx_17 = split_17[1]
        gates_17 = split_17[2]
        split_17 = None
        input_137 = torch.conv2d(
            ctx_17,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_17 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_138 = torch._C._nn.gelu(input_137, approximate="none")
        input_137 = None
        getitem_139 = gates_17[(slice(None, None, None), slice(0, 1, None))]
        mul_136 = input_138 * getitem_139
        getitem_139 = None
        ctx_all_85 = 0 + mul_136
        mul_136 = None
        input_139 = torch.conv2d(
            input_138,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_138 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_140 = torch._C._nn.gelu(input_139, approximate="none")
        input_139 = None
        getitem_140 = gates_17[(slice(None, None, None), slice(1, 2, None))]
        mul_137 = input_140 * getitem_140
        getitem_140 = None
        ctx_all_86 = ctx_all_85 + mul_137
        ctx_all_85 = mul_137 = None
        input_141 = torch.conv2d(
            input_140,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_140 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_142 = torch._C._nn.gelu(input_141, approximate="none")
        input_141 = None
        getitem_141 = gates_17[(slice(None, None, None), slice(2, 3, None))]
        mul_138 = input_142 * getitem_141
        getitem_141 = None
        ctx_all_87 = ctx_all_86 + mul_138
        ctx_all_86 = mul_138 = None
        input_143 = torch.conv2d(
            input_142,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_142 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_144 = torch._C._nn.gelu(input_143, approximate="none")
        input_143 = None
        getitem_142 = gates_17[(slice(None, None, None), slice(3, 4, None))]
        mul_139 = input_144 * getitem_142
        getitem_142 = None
        ctx_all_88 = ctx_all_87 + mul_139
        ctx_all_87 = mul_139 = None
        mean_17 = input_144.mean((2, 3), keepdim=True)
        input_144 = None
        ctx_global_17 = torch._C._nn.gelu(mean_17, approximate="none")
        mean_17 = None
        getitem_143 = gates_17[(slice(None, None, None), slice(4, None, None))]
        gates_17 = None
        mul_140 = ctx_global_17 * getitem_143
        ctx_global_17 = getitem_143 = None
        ctx_all_89 = ctx_all_88 + mul_140
        ctx_all_88 = mul_140 = None
        conv2d_161 = torch.conv2d(
            ctx_all_89,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_89 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_51 = q_17 * conv2d_161
        q_17 = conv2d_161 = None
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
        x_251 = x_out_53.permute(0, 2, 3, 1)
        x_out_53 = None
        x_252 = torch.nn.functional.layer_norm(
            x_251,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_251 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_post_parameters_bias_ = (None)
        x_253 = x_252.permute(0, 3, 1, 2)
        x_252 = None
        gamma_34 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_142 = x_253 * gamma_34
        x_253 = gamma_34 = None
        x_254 = x_249 + mul_142
        x_249 = mul_142 = None
        x_255 = torch.conv2d(
            x_254,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_256 = torch._C._nn.gelu(x_255, approximate="none")
        x_255 = None
        x_257 = torch.nn.functional.dropout(x_256, 0.0, False, False)
        x_256 = None
        x_258 = torch.conv2d(
            x_257,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_257 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_259 = torch.nn.functional.dropout(x_258, 0.0, False, False)
        x_258 = None
        x_260 = x_259.permute(0, 2, 3, 1)
        x_259 = None
        x_261 = torch.nn.functional.layer_norm(
            x_260,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_260 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_post_parameters_bias_ = (None)
        x_262 = x_261.permute(0, 3, 1, 2)
        x_261 = None
        gamma_35 = l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_13_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_143 = x_262 * gamma_35
        x_262 = gamma_35 = None
        x_263 = x_254 + mul_143
        x_254 = mul_143 = None
        x_264 = torch.conv2d(
            x_263,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_bias_ = (None)
        split_18 = torch.functional.split(x_264, [768, 768, 5], 1)
        x_264 = None
        q_18 = split_18[0]
        ctx_18 = split_18[1]
        gates_18 = split_18[2]
        split_18 = None
        input_145 = torch.conv2d(
            ctx_18,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_18 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_146 = torch._C._nn.gelu(input_145, approximate="none")
        input_145 = None
        getitem_147 = gates_18[(slice(None, None, None), slice(0, 1, None))]
        mul_144 = input_146 * getitem_147
        getitem_147 = None
        ctx_all_90 = 0 + mul_144
        mul_144 = None
        input_147 = torch.conv2d(
            input_146,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_146 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_148 = torch._C._nn.gelu(input_147, approximate="none")
        input_147 = None
        getitem_148 = gates_18[(slice(None, None, None), slice(1, 2, None))]
        mul_145 = input_148 * getitem_148
        getitem_148 = None
        ctx_all_91 = ctx_all_90 + mul_145
        ctx_all_90 = mul_145 = None
        input_149 = torch.conv2d(
            input_148,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_148 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_150 = torch._C._nn.gelu(input_149, approximate="none")
        input_149 = None
        getitem_149 = gates_18[(slice(None, None, None), slice(2, 3, None))]
        mul_146 = input_150 * getitem_149
        getitem_149 = None
        ctx_all_92 = ctx_all_91 + mul_146
        ctx_all_91 = mul_146 = None
        input_151 = torch.conv2d(
            input_150,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_150 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_152 = torch._C._nn.gelu(input_151, approximate="none")
        input_151 = None
        getitem_150 = gates_18[(slice(None, None, None), slice(3, 4, None))]
        mul_147 = input_152 * getitem_150
        getitem_150 = None
        ctx_all_93 = ctx_all_92 + mul_147
        ctx_all_92 = mul_147 = None
        mean_18 = input_152.mean((2, 3), keepdim=True)
        input_152 = None
        ctx_global_18 = torch._C._nn.gelu(mean_18, approximate="none")
        mean_18 = None
        getitem_151 = gates_18[(slice(None, None, None), slice(4, None, None))]
        gates_18 = None
        mul_148 = ctx_global_18 * getitem_151
        ctx_global_18 = getitem_151 = None
        ctx_all_94 = ctx_all_93 + mul_148
        ctx_all_93 = mul_148 = None
        conv2d_170 = torch.conv2d(
            ctx_all_94,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_94 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_54 = q_18 * conv2d_170
        q_18 = conv2d_170 = None
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
        x_265 = x_out_56.permute(0, 2, 3, 1)
        x_out_56 = None
        x_266 = torch.nn.functional.layer_norm(
            x_265,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_265 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_post_parameters_bias_ = (None)
        x_267 = x_266.permute(0, 3, 1, 2)
        x_266 = None
        gamma_36 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_150 = x_267 * gamma_36
        x_267 = gamma_36 = None
        x_268 = x_263 + mul_150
        x_263 = mul_150 = None
        x_269 = torch.conv2d(
            x_268,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_270 = torch._C._nn.gelu(x_269, approximate="none")
        x_269 = None
        x_271 = torch.nn.functional.dropout(x_270, 0.0, False, False)
        x_270 = None
        x_272 = torch.conv2d(
            x_271,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_271 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_273 = torch.nn.functional.dropout(x_272, 0.0, False, False)
        x_272 = None
        x_274 = x_273.permute(0, 2, 3, 1)
        x_273 = None
        x_275 = torch.nn.functional.layer_norm(
            x_274,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_274 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_post_parameters_bias_ = (None)
        x_276 = x_275.permute(0, 3, 1, 2)
        x_275 = None
        gamma_37 = l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_14_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_151 = x_276 * gamma_37
        x_276 = gamma_37 = None
        x_277 = x_268 + mul_151
        x_268 = mul_151 = None
        x_278 = torch.conv2d(
            x_277,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_bias_ = (None)
        split_19 = torch.functional.split(x_278, [768, 768, 5], 1)
        x_278 = None
        q_19 = split_19[0]
        ctx_19 = split_19[1]
        gates_19 = split_19[2]
        split_19 = None
        input_153 = torch.conv2d(
            ctx_19,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_19 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_154 = torch._C._nn.gelu(input_153, approximate="none")
        input_153 = None
        getitem_155 = gates_19[(slice(None, None, None), slice(0, 1, None))]
        mul_152 = input_154 * getitem_155
        getitem_155 = None
        ctx_all_95 = 0 + mul_152
        mul_152 = None
        input_155 = torch.conv2d(
            input_154,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_154 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_156 = torch._C._nn.gelu(input_155, approximate="none")
        input_155 = None
        getitem_156 = gates_19[(slice(None, None, None), slice(1, 2, None))]
        mul_153 = input_156 * getitem_156
        getitem_156 = None
        ctx_all_96 = ctx_all_95 + mul_153
        ctx_all_95 = mul_153 = None
        input_157 = torch.conv2d(
            input_156,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_156 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_158 = torch._C._nn.gelu(input_157, approximate="none")
        input_157 = None
        getitem_157 = gates_19[(slice(None, None, None), slice(2, 3, None))]
        mul_154 = input_158 * getitem_157
        getitem_157 = None
        ctx_all_97 = ctx_all_96 + mul_154
        ctx_all_96 = mul_154 = None
        input_159 = torch.conv2d(
            input_158,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_158 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_160 = torch._C._nn.gelu(input_159, approximate="none")
        input_159 = None
        getitem_158 = gates_19[(slice(None, None, None), slice(3, 4, None))]
        mul_155 = input_160 * getitem_158
        getitem_158 = None
        ctx_all_98 = ctx_all_97 + mul_155
        ctx_all_97 = mul_155 = None
        mean_19 = input_160.mean((2, 3), keepdim=True)
        input_160 = None
        ctx_global_19 = torch._C._nn.gelu(mean_19, approximate="none")
        mean_19 = None
        getitem_159 = gates_19[(slice(None, None, None), slice(4, None, None))]
        gates_19 = None
        mul_156 = ctx_global_19 * getitem_159
        ctx_global_19 = getitem_159 = None
        ctx_all_99 = ctx_all_98 + mul_156
        ctx_all_98 = mul_156 = None
        conv2d_179 = torch.conv2d(
            ctx_all_99,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_99 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_57 = q_19 * conv2d_179
        q_19 = conv2d_179 = None
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
        x_279 = x_out_59.permute(0, 2, 3, 1)
        x_out_59 = None
        x_280 = torch.nn.functional.layer_norm(
            x_279,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_279 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_post_parameters_bias_ = (None)
        x_281 = x_280.permute(0, 3, 1, 2)
        x_280 = None
        gamma_38 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_158 = x_281 * gamma_38
        x_281 = gamma_38 = None
        x_282 = x_277 + mul_158
        x_277 = mul_158 = None
        x_283 = torch.conv2d(
            x_282,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_284 = torch._C._nn.gelu(x_283, approximate="none")
        x_283 = None
        x_285 = torch.nn.functional.dropout(x_284, 0.0, False, False)
        x_284 = None
        x_286 = torch.conv2d(
            x_285,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_285 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_287 = torch.nn.functional.dropout(x_286, 0.0, False, False)
        x_286 = None
        x_288 = x_287.permute(0, 2, 3, 1)
        x_287 = None
        x_289 = torch.nn.functional.layer_norm(
            x_288,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_288 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_post_parameters_bias_ = (None)
        x_290 = x_289.permute(0, 3, 1, 2)
        x_289 = None
        gamma_39 = l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_15_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_159 = x_290 * gamma_39
        x_290 = gamma_39 = None
        x_291 = x_282 + mul_159
        x_282 = mul_159 = None
        x_292 = torch.conv2d(
            x_291,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_bias_ = (None)
        split_20 = torch.functional.split(x_292, [768, 768, 5], 1)
        x_292 = None
        q_20 = split_20[0]
        ctx_20 = split_20[1]
        gates_20 = split_20[2]
        split_20 = None
        input_161 = torch.conv2d(
            ctx_20,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_20 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_162 = torch._C._nn.gelu(input_161, approximate="none")
        input_161 = None
        getitem_163 = gates_20[(slice(None, None, None), slice(0, 1, None))]
        mul_160 = input_162 * getitem_163
        getitem_163 = None
        ctx_all_100 = 0 + mul_160
        mul_160 = None
        input_163 = torch.conv2d(
            input_162,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_162 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_164 = torch._C._nn.gelu(input_163, approximate="none")
        input_163 = None
        getitem_164 = gates_20[(slice(None, None, None), slice(1, 2, None))]
        mul_161 = input_164 * getitem_164
        getitem_164 = None
        ctx_all_101 = ctx_all_100 + mul_161
        ctx_all_100 = mul_161 = None
        input_165 = torch.conv2d(
            input_164,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_164 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_166 = torch._C._nn.gelu(input_165, approximate="none")
        input_165 = None
        getitem_165 = gates_20[(slice(None, None, None), slice(2, 3, None))]
        mul_162 = input_166 * getitem_165
        getitem_165 = None
        ctx_all_102 = ctx_all_101 + mul_162
        ctx_all_101 = mul_162 = None
        input_167 = torch.conv2d(
            input_166,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_166 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_168 = torch._C._nn.gelu(input_167, approximate="none")
        input_167 = None
        getitem_166 = gates_20[(slice(None, None, None), slice(3, 4, None))]
        mul_163 = input_168 * getitem_166
        getitem_166 = None
        ctx_all_103 = ctx_all_102 + mul_163
        ctx_all_102 = mul_163 = None
        mean_20 = input_168.mean((2, 3), keepdim=True)
        input_168 = None
        ctx_global_20 = torch._C._nn.gelu(mean_20, approximate="none")
        mean_20 = None
        getitem_167 = gates_20[(slice(None, None, None), slice(4, None, None))]
        gates_20 = None
        mul_164 = ctx_global_20 * getitem_167
        ctx_global_20 = getitem_167 = None
        ctx_all_104 = ctx_all_103 + mul_164
        ctx_all_103 = mul_164 = None
        conv2d_188 = torch.conv2d(
            ctx_all_104,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_104 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_60 = q_20 * conv2d_188
        q_20 = conv2d_188 = None
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
        x_293 = x_out_62.permute(0, 2, 3, 1)
        x_out_62 = None
        x_294 = torch.nn.functional.layer_norm(
            x_293,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_293 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_post_parameters_bias_ = (None)
        x_295 = x_294.permute(0, 3, 1, 2)
        x_294 = None
        gamma_40 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_166 = x_295 * gamma_40
        x_295 = gamma_40 = None
        x_296 = x_291 + mul_166
        x_291 = mul_166 = None
        x_297 = torch.conv2d(
            x_296,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_298 = torch._C._nn.gelu(x_297, approximate="none")
        x_297 = None
        x_299 = torch.nn.functional.dropout(x_298, 0.0, False, False)
        x_298 = None
        x_300 = torch.conv2d(
            x_299,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_299 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_301 = torch.nn.functional.dropout(x_300, 0.0, False, False)
        x_300 = None
        x_302 = x_301.permute(0, 2, 3, 1)
        x_301 = None
        x_303 = torch.nn.functional.layer_norm(
            x_302,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_302 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_post_parameters_bias_ = (None)
        x_304 = x_303.permute(0, 3, 1, 2)
        x_303 = None
        gamma_41 = l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_16_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_167 = x_304 * gamma_41
        x_304 = gamma_41 = None
        x_305 = x_296 + mul_167
        x_296 = mul_167 = None
        x_306 = torch.conv2d(
            x_305,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_bias_ = (None)
        split_21 = torch.functional.split(x_306, [768, 768, 5], 1)
        x_306 = None
        q_21 = split_21[0]
        ctx_21 = split_21[1]
        gates_21 = split_21[2]
        split_21 = None
        input_169 = torch.conv2d(
            ctx_21,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        ctx_21 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_170 = torch._C._nn.gelu(input_169, approximate="none")
        input_169 = None
        getitem_171 = gates_21[(slice(None, None, None), slice(0, 1, None))]
        mul_168 = input_170 * getitem_171
        getitem_171 = None
        ctx_all_105 = 0 + mul_168
        mul_168 = None
        input_171 = torch.conv2d(
            input_170,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        input_170 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_172 = torch._C._nn.gelu(input_171, approximate="none")
        input_171 = None
        getitem_172 = gates_21[(slice(None, None, None), slice(1, 2, None))]
        mul_169 = input_172 * getitem_172
        getitem_172 = None
        ctx_all_106 = ctx_all_105 + mul_169
        ctx_all_105 = mul_169 = None
        input_173 = torch.conv2d(
            input_172,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        input_172 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_174 = torch._C._nn.gelu(input_173, approximate="none")
        input_173 = None
        getitem_173 = gates_21[(slice(None, None, None), slice(2, 3, None))]
        mul_170 = input_174 * getitem_173
        getitem_173 = None
        ctx_all_107 = ctx_all_106 + mul_170
        ctx_all_106 = mul_170 = None
        input_175 = torch.conv2d(
            input_174,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            768,
        )
        input_174 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_176 = torch._C._nn.gelu(input_175, approximate="none")
        input_175 = None
        getitem_174 = gates_21[(slice(None, None, None), slice(3, 4, None))]
        mul_171 = input_176 * getitem_174
        getitem_174 = None
        ctx_all_108 = ctx_all_107 + mul_171
        ctx_all_107 = mul_171 = None
        mean_21 = input_176.mean((2, 3), keepdim=True)
        input_176 = None
        ctx_global_21 = torch._C._nn.gelu(mean_21, approximate="none")
        mean_21 = None
        getitem_175 = gates_21[(slice(None, None, None), slice(4, None, None))]
        gates_21 = None
        mul_172 = ctx_global_21 * getitem_175
        ctx_global_21 = getitem_175 = None
        ctx_all_109 = ctx_all_108 + mul_172
        ctx_all_108 = mul_172 = None
        conv2d_197 = torch.conv2d(
            ctx_all_109,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_109 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_63 = q_21 * conv2d_197
        q_21 = conv2d_197 = None
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
        x_307 = x_out_65.permute(0, 2, 3, 1)
        x_out_65 = None
        x_308 = torch.nn.functional.layer_norm(
            x_307,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_307 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_post_parameters_bias_ = (None)
        x_309 = x_308.permute(0, 3, 1, 2)
        x_308 = None
        gamma_42 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_174 = x_309 * gamma_42
        x_309 = gamma_42 = None
        x_310 = x_305 + mul_174
        x_305 = mul_174 = None
        x_311 = torch.conv2d(
            x_310,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_312 = torch._C._nn.gelu(x_311, approximate="none")
        x_311 = None
        x_313 = torch.nn.functional.dropout(x_312, 0.0, False, False)
        x_312 = None
        x_314 = torch.conv2d(
            x_313,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_313 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_315 = torch.nn.functional.dropout(x_314, 0.0, False, False)
        x_314 = None
        x_316 = x_315.permute(0, 2, 3, 1)
        x_315 = None
        x_317 = torch.nn.functional.layer_norm(
            x_316,
            (768,),
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_316 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_post_parameters_bias_ = (None)
        x_318 = x_317.permute(0, 3, 1, 2)
        x_317 = None
        gamma_43 = l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_2_modules_blocks_modules_17_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_175 = x_318 * gamma_43
        x_318 = gamma_43 = None
        x_319 = x_310 + mul_175
        x_310 = mul_175 = None
        x_320 = torch.conv2d(
            x_319,
            l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_319 = l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_ = (None)
        x_321 = x_320.permute(0, 2, 3, 1)
        x_320 = None
        x_322 = torch.nn.functional.layer_norm(
            x_321,
            (1536,),
            l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_,
            l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_,
            1e-05,
        )
        x_321 = l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_ = l_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_ = (None)
        x_323 = x_322.permute(0, 3, 1, 2)
        x_322 = None
        x_324 = torch.conv2d(
            x_323,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_ = (None)
        split_22 = torch.functional.split(x_324, [1536, 1536, 5], 1)
        x_324 = None
        q_22 = split_22[0]
        ctx_22 = split_22[1]
        gates_22 = split_22[2]
        split_22 = None
        input_177 = torch.conv2d(
            ctx_22,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        ctx_22 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_178 = torch._C._nn.gelu(input_177, approximate="none")
        input_177 = None
        getitem_179 = gates_22[(slice(None, None, None), slice(0, 1, None))]
        mul_176 = input_178 * getitem_179
        getitem_179 = None
        ctx_all_110 = 0 + mul_176
        mul_176 = None
        input_179 = torch.conv2d(
            input_178,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1536,
        )
        input_178 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_180 = torch._C._nn.gelu(input_179, approximate="none")
        input_179 = None
        getitem_180 = gates_22[(slice(None, None, None), slice(1, 2, None))]
        mul_177 = input_180 * getitem_180
        getitem_180 = None
        ctx_all_111 = ctx_all_110 + mul_177
        ctx_all_110 = mul_177 = None
        input_181 = torch.conv2d(
            input_180,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            1536,
        )
        input_180 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_182 = torch._C._nn.gelu(input_181, approximate="none")
        input_181 = None
        getitem_181 = gates_22[(slice(None, None, None), slice(2, 3, None))]
        mul_178 = input_182 * getitem_181
        getitem_181 = None
        ctx_all_112 = ctx_all_111 + mul_178
        ctx_all_111 = mul_178 = None
        input_183 = torch.conv2d(
            input_182,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            1536,
        )
        input_182 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_184 = torch._C._nn.gelu(input_183, approximate="none")
        input_183 = None
        getitem_182 = gates_22[(slice(None, None, None), slice(3, 4, None))]
        mul_179 = input_184 * getitem_182
        getitem_182 = None
        ctx_all_113 = ctx_all_112 + mul_179
        ctx_all_112 = mul_179 = None
        mean_22 = input_184.mean((2, 3), keepdim=True)
        input_184 = None
        ctx_global_22 = torch._C._nn.gelu(mean_22, approximate="none")
        mean_22 = None
        getitem_183 = gates_22[(slice(None, None, None), slice(4, None, None))]
        gates_22 = None
        mul_180 = ctx_global_22 * getitem_183
        ctx_global_22 = getitem_183 = None
        ctx_all_114 = ctx_all_113 + mul_180
        ctx_all_113 = mul_180 = None
        conv2d_207 = torch.conv2d(
            ctx_all_114,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_114 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_66 = q_22 * conv2d_207
        q_22 = conv2d_207 = None
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
        x_325 = x_out_68.permute(0, 2, 3, 1)
        x_out_68 = None
        x_326 = torch.nn.functional.layer_norm(
            x_325,
            (1536,),
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_325 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_post_parameters_bias_ = (None)
        x_327 = x_326.permute(0, 3, 1, 2)
        x_326 = None
        gamma_44 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_182 = x_327 * gamma_44
        x_327 = gamma_44 = None
        x_328 = x_323 + mul_182
        x_323 = mul_182 = None
        x_329 = torch.conv2d(
            x_328,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_330 = torch._C._nn.gelu(x_329, approximate="none")
        x_329 = None
        x_331 = torch.nn.functional.dropout(x_330, 0.0, False, False)
        x_330 = None
        x_332 = torch.conv2d(
            x_331,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_331 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_333 = torch.nn.functional.dropout(x_332, 0.0, False, False)
        x_332 = None
        x_334 = x_333.permute(0, 2, 3, 1)
        x_333 = None
        x_335 = torch.nn.functional.layer_norm(
            x_334,
            (1536,),
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_334 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_post_parameters_bias_ = (None)
        x_336 = x_335.permute(0, 3, 1, 2)
        x_335 = None
        gamma_45 = l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_3_modules_blocks_modules_0_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_183 = x_336 * gamma_45
        x_336 = gamma_45 = None
        x_337 = x_328 + mul_183
        x_328 = mul_183 = None
        x_338 = torch.conv2d(
            x_337,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_ = (None)
        split_23 = torch.functional.split(x_338, [1536, 1536, 5], 1)
        x_338 = None
        q_23 = split_23[0]
        ctx_23 = split_23[1]
        gates_23 = split_23[2]
        split_23 = None
        input_185 = torch.conv2d(
            ctx_23,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        ctx_23 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_ = (None)
        input_186 = torch._C._nn.gelu(input_185, approximate="none")
        input_185 = None
        getitem_187 = gates_23[(slice(None, None, None), slice(0, 1, None))]
        mul_184 = input_186 * getitem_187
        getitem_187 = None
        ctx_all_115 = 0 + mul_184
        mul_184 = None
        input_187 = torch.conv2d(
            input_186,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1536,
        )
        input_186 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_ = (None)
        input_188 = torch._C._nn.gelu(input_187, approximate="none")
        input_187 = None
        getitem_188 = gates_23[(slice(None, None, None), slice(1, 2, None))]
        mul_185 = input_188 * getitem_188
        getitem_188 = None
        ctx_all_116 = ctx_all_115 + mul_185
        ctx_all_115 = mul_185 = None
        input_189 = torch.conv2d(
            input_188,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            1536,
        )
        input_188 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_ = (None)
        input_190 = torch._C._nn.gelu(input_189, approximate="none")
        input_189 = None
        getitem_189 = gates_23[(slice(None, None, None), slice(2, 3, None))]
        mul_186 = input_190 * getitem_189
        getitem_189 = None
        ctx_all_117 = ctx_all_116 + mul_186
        ctx_all_116 = mul_186 = None
        input_191 = torch.conv2d(
            input_190,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            1536,
        )
        input_190 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_ = (None)
        input_192 = torch._C._nn.gelu(input_191, approximate="none")
        input_191 = None
        getitem_190 = gates_23[(slice(None, None, None), slice(3, 4, None))]
        mul_187 = input_192 * getitem_190
        getitem_190 = None
        ctx_all_118 = ctx_all_117 + mul_187
        ctx_all_117 = mul_187 = None
        mean_23 = input_192.mean((2, 3), keepdim=True)
        input_192 = None
        ctx_global_23 = torch._C._nn.gelu(mean_23, approximate="none")
        mean_23 = None
        getitem_191 = gates_23[(slice(None, None, None), slice(4, None, None))]
        gates_23 = None
        mul_188 = ctx_global_23 * getitem_191
        ctx_global_23 = getitem_191 = None
        ctx_all_119 = ctx_all_118 + mul_188
        ctx_all_118 = mul_188 = None
        conv2d_216 = torch.conv2d(
            ctx_all_119,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        ctx_all_119 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_ = (None)
        x_out_69 = q_23 * conv2d_216
        q_23 = conv2d_216 = None
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
        x_339 = x_out_71.permute(0, 2, 3, 1)
        x_out_71 = None
        x_340 = torch.nn.functional.layer_norm(
            x_339,
            (1536,),
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_post_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_post_parameters_bias_,
            1e-05,
        )
        x_339 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_post_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_post_parameters_bias_ = (None)
        x_341 = x_340.permute(0, 3, 1, 2)
        x_340 = None
        gamma_46 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_ls1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_ls1_parameters_gamma_ = (
            None
        )
        mul_190 = x_341 * gamma_46
        x_341 = gamma_46 = None
        x_342 = x_337 + mul_190
        x_337 = mul_190 = None
        x_343 = torch.conv2d(
            x_342,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_344 = torch._C._nn.gelu(x_343, approximate="none")
        x_343 = None
        x_345 = torch.nn.functional.dropout(x_344, 0.0, False, False)
        x_344 = None
        x_346 = torch.conv2d(
            x_345,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_345 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_347 = torch.nn.functional.dropout(x_346, 0.0, False, False)
        x_346 = None
        x_348 = x_347.permute(0, 2, 3, 1)
        x_347 = None
        x_349 = torch.nn.functional.layer_norm(
            x_348,
            (1536,),
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_post_parameters_weight_,
            l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_post_parameters_bias_,
            1e-05,
        )
        x_348 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_post_parameters_weight_ = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_post_parameters_bias_ = (None)
        x_350 = x_349.permute(0, 3, 1, 2)
        x_349 = None
        gamma_47 = l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_ls2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_layers_modules_3_modules_blocks_modules_1_modules_ls2_parameters_gamma_ = (
            None
        )
        mul_191 = x_350 * gamma_47
        x_350 = gamma_47 = None
        x_351 = x_342 + mul_191
        x_342 = mul_191 = None
        x_352 = x_351.permute(0, 2, 3, 1)
        x_351 = None
        x_353 = torch.nn.functional.layer_norm(
            x_352,
            (1536,),
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
