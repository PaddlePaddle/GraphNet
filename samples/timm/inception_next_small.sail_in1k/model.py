import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_downsample_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_downsample_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_12_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_13_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_14_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_15_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_16_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_17_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_18_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_19_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_20_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_21_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_22_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_23_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_24_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_25_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_26_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_0_parameters_weight_ = (
            L_self_modules_stem_modules_0_parameters_weight_
        )
        l_self_modules_stem_modules_0_parameters_bias_ = (
            L_self_modules_stem_modules_0_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_1_buffers_running_mean_ = (
            L_self_modules_stem_modules_1_buffers_running_mean_
        )
        l_self_modules_stem_modules_1_buffers_running_var_ = (
            L_self_modules_stem_modules_1_buffers_running_var_
        )
        l_self_modules_stem_modules_1_parameters_weight_ = (
            L_self_modules_stem_modules_1_parameters_weight_
        )
        l_self_modules_stem_modules_1_parameters_bias_ = (
            L_self_modules_stem_modules_1_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_ = (
            L_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_ = (
            L_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_parameters_gamma_ = (
            L_self_modules_stages_modules_0_modules_blocks_modules_2_parameters_gamma_
        )
        l_self_modules_stages_modules_1_modules_downsample_modules_0_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_downsample_modules_0_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_downsample_modules_0_buffers_running_var_ = L_self_modules_stages_modules_1_modules_downsample_modules_0_buffers_running_var_
        l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_
        l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_ = (
            L_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_ = (
            L_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_parameters_gamma_ = (
            L_self_modules_stages_modules_1_modules_blocks_modules_2_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_downsample_modules_0_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_downsample_modules_0_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_downsample_modules_0_buffers_running_var_ = L_self_modules_stages_modules_2_modules_downsample_modules_0_buffers_running_var_
        l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_
        l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_4_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_5_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_6_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_7_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_8_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_9_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_10_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_11_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_12_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_12_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_13_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_13_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_14_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_14_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_15_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_15_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_16_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_16_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_17_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_17_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_18_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_18_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_19_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_19_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_20_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_20_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_21_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_21_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_22_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_22_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_23_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_23_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_24_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_24_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_25_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_25_parameters_gamma_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_26_parameters_gamma_ = (
            L_self_modules_stages_modules_2_modules_blocks_modules_26_parameters_gamma_
        )
        l_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_var_ = L_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_var_
        l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_
        l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_ = (
            L_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_ = (
            L_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_parameters_gamma_ = (
            L_self_modules_stages_modules_3_modules_blocks_modules_2_parameters_gamma_
        )
        l_self_modules_head_modules_fc1_parameters_weight_ = (
            L_self_modules_head_modules_fc1_parameters_weight_
        )
        l_self_modules_head_modules_fc1_parameters_bias_ = (
            L_self_modules_head_modules_fc1_parameters_bias_
        )
        l_self_modules_head_modules_norm_parameters_weight_ = (
            L_self_modules_head_modules_norm_parameters_weight_
        )
        l_self_modules_head_modules_norm_parameters_bias_ = (
            L_self_modules_head_modules_norm_parameters_bias_
        )
        l_self_modules_head_modules_fc2_parameters_weight_ = (
            L_self_modules_head_modules_fc2_parameters_weight_
        )
        l_self_modules_head_modules_fc2_parameters_bias_ = (
            L_self_modules_head_modules_fc2_parameters_bias_
        )
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_0_parameters_weight_,
            l_self_modules_stem_modules_0_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_stem_modules_0_parameters_weight_
        ) = l_self_modules_stem_modules_0_parameters_bias_ = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_stem_modules_1_buffers_running_mean_,
            l_self_modules_stem_modules_1_buffers_running_var_,
            l_self_modules_stem_modules_1_parameters_weight_,
            l_self_modules_stem_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_stem_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_1_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_1_parameters_weight_
        ) = l_self_modules_stem_modules_1_parameters_bias_ = None
        split = torch.functional.split(input_2, (60, 12, 12, 12), dim=1)
        x_id = split[0]
        x_hw = split[1]
        x_w = split[2]
        x_h = split[3]
        split = None
        conv2d_1 = torch.conv2d(
            x_hw,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            12,
        )
        x_hw = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_2 = torch.conv2d(
            x_w,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            12,
        )
        x_w = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_3 = torch.conv2d(
            x_h,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            12,
        )
        x_h = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x = torch.cat((x_id, conv2d_1, conv2d_2, conv2d_3), dim=1)
        x_id = conv2d_1 = conv2d_2 = conv2d_3 = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_2 = torch.conv2d(
            x_1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_1 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
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
        reshape = l_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_ = (
            None
        )
        x_6 = x_5.mul(reshape)
        x_5 = reshape = None
        x_7 = x_6 + input_2
        x_6 = input_2 = None
        split_1 = torch.functional.split(x_7, (60, 12, 12, 12), dim=1)
        x_id_1 = split_1[0]
        x_hw_1 = split_1[1]
        x_w_1 = split_1[2]
        x_h_1 = split_1[3]
        split_1 = None
        conv2d_6 = torch.conv2d(
            x_hw_1,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            12,
        )
        x_hw_1 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_7 = torch.conv2d(
            x_w_1,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            12,
        )
        x_w_1 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_8 = torch.conv2d(
            x_h_1,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            12,
        )
        x_h_1 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_8 = torch.cat((x_id_1, conv2d_6, conv2d_7, conv2d_8), dim=1)
        x_id_1 = conv2d_6 = conv2d_7 = conv2d_8 = None
        x_9 = torch.nn.functional.batch_norm(
            x_8,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_8 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_11 = torch._C._nn.gelu(x_10, approximate="none")
        x_10 = None
        x_12 = torch.nn.functional.dropout(x_11, 0.0, False, False)
        x_11 = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_12 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_1 = l_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_ = (
            None
        )
        x_14 = x_13.mul(reshape_1)
        x_13 = reshape_1 = None
        x_15 = x_14 + x_7
        x_14 = x_7 = None
        split_2 = torch.functional.split(x_15, (60, 12, 12, 12), dim=1)
        x_id_2 = split_2[0]
        x_hw_2 = split_2[1]
        x_w_2 = split_2[2]
        x_h_2 = split_2[3]
        split_2 = None
        conv2d_11 = torch.conv2d(
            x_hw_2,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            12,
        )
        x_hw_2 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_12 = torch.conv2d(
            x_w_2,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            12,
        )
        x_w_2 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_13 = torch.conv2d(
            x_h_2,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            12,
        )
        x_h_2 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_16 = torch.cat((x_id_2, conv2d_11, conv2d_12, conv2d_13), dim=1)
        x_id_2 = conv2d_11 = conv2d_12 = conv2d_13 = None
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_19 = torch._C._nn.gelu(x_18, approximate="none")
        x_18 = None
        x_20 = torch.nn.functional.dropout(x_19, 0.0, False, False)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_2 = l_self_modules_stages_modules_0_modules_blocks_modules_2_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_parameters_gamma_ = (
            None
        )
        x_22 = x_21.mul(reshape_2)
        x_21 = reshape_2 = None
        x_23 = x_22 + x_15
        x_22 = x_15 = None
        input_3 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_stages_modules_1_modules_downsample_modules_0_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_downsample_modules_0_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_stages_modules_1_modules_downsample_modules_0_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_downsample_modules_0_buffers_running_var_ = l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_ = (None)
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_ = (None)
        split_3 = torch.functional.split(input_4, (120, 24, 24, 24), dim=1)
        x_id_3 = split_3[0]
        x_hw_3 = split_3[1]
        x_w_3 = split_3[2]
        x_h_3 = split_3[3]
        split_3 = None
        conv2d_17 = torch.conv2d(
            x_hw_3,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        x_hw_3 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_18 = torch.conv2d(
            x_w_3,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            24,
        )
        x_w_3 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_19 = torch.conv2d(
            x_h_3,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            24,
        )
        x_h_3 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_24 = torch.cat((x_id_3, conv2d_17, conv2d_18, conv2d_19), dim=1)
        x_id_3 = conv2d_17 = conv2d_18 = conv2d_19 = None
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_27 = torch._C._nn.gelu(x_26, approximate="none")
        x_26 = None
        x_28 = torch.nn.functional.dropout(x_27, 0.0, False, False)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_3 = l_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_ = (
            None
        )
        x_30 = x_29.mul(reshape_3)
        x_29 = reshape_3 = None
        x_31 = x_30 + input_4
        x_30 = input_4 = None
        split_4 = torch.functional.split(x_31, (120, 24, 24, 24), dim=1)
        x_id_4 = split_4[0]
        x_hw_4 = split_4[1]
        x_w_4 = split_4[2]
        x_h_4 = split_4[3]
        split_4 = None
        conv2d_22 = torch.conv2d(
            x_hw_4,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        x_hw_4 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_23 = torch.conv2d(
            x_w_4,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            24,
        )
        x_w_4 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_24 = torch.conv2d(
            x_h_4,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            24,
        )
        x_h_4 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_32 = torch.cat((x_id_4, conv2d_22, conv2d_23, conv2d_24), dim=1)
        x_id_4 = conv2d_22 = conv2d_23 = conv2d_24 = None
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_35 = torch._C._nn.gelu(x_34, approximate="none")
        x_34 = None
        x_36 = torch.nn.functional.dropout(x_35, 0.0, False, False)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_36 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_4 = l_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_ = (
            None
        )
        x_38 = x_37.mul(reshape_4)
        x_37 = reshape_4 = None
        x_39 = x_38 + x_31
        x_38 = x_31 = None
        split_5 = torch.functional.split(x_39, (120, 24, 24, 24), dim=1)
        x_id_5 = split_5[0]
        x_hw_5 = split_5[1]
        x_w_5 = split_5[2]
        x_h_5 = split_5[3]
        split_5 = None
        conv2d_27 = torch.conv2d(
            x_hw_5,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        x_hw_5 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_28 = torch.conv2d(
            x_w_5,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            24,
        )
        x_w_5 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_29 = torch.conv2d(
            x_h_5,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            24,
        )
        x_h_5 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_40 = torch.cat((x_id_5, conv2d_27, conv2d_28, conv2d_29), dim=1)
        x_id_5 = conv2d_27 = conv2d_28 = conv2d_29 = None
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_41 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_43 = torch._C._nn.gelu(x_42, approximate="none")
        x_42 = None
        x_44 = torch.nn.functional.dropout(x_43, 0.0, False, False)
        x_43 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_44 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_5 = l_self_modules_stages_modules_1_modules_blocks_modules_2_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_parameters_gamma_ = (
            None
        )
        x_46 = x_45.mul(reshape_5)
        x_45 = reshape_5 = None
        x_47 = x_46 + x_39
        x_46 = x_39 = None
        input_5 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_stages_modules_2_modules_downsample_modules_0_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_downsample_modules_0_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_stages_modules_2_modules_downsample_modules_0_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_downsample_modules_0_buffers_running_var_ = l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_ = (None)
        input_6 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_ = (None)
        split_6 = torch.functional.split(input_6, (240, 48, 48, 48), dim=1)
        x_id_6 = split_6[0]
        x_hw_6 = split_6[1]
        x_w_6 = split_6[2]
        x_h_6 = split_6[3]
        split_6 = None
        conv2d_33 = torch.conv2d(
            x_hw_6,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_6 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_34 = torch.conv2d(
            x_w_6,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_6 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_35 = torch.conv2d(
            x_h_6,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_6 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_48 = torch.cat((x_id_6, conv2d_33, conv2d_34, conv2d_35), dim=1)
        x_id_6 = conv2d_33 = conv2d_34 = conv2d_35 = None
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_51 = torch._C._nn.gelu(x_50, approximate="none")
        x_50 = None
        x_52 = torch.nn.functional.dropout(x_51, 0.0, False, False)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_6 = l_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_ = (
            None
        )
        x_54 = x_53.mul(reshape_6)
        x_53 = reshape_6 = None
        x_55 = x_54 + input_6
        x_54 = input_6 = None
        split_7 = torch.functional.split(x_55, (240, 48, 48, 48), dim=1)
        x_id_7 = split_7[0]
        x_hw_7 = split_7[1]
        x_w_7 = split_7[2]
        x_h_7 = split_7[3]
        split_7 = None
        conv2d_38 = torch.conv2d(
            x_hw_7,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_7 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_39 = torch.conv2d(
            x_w_7,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_7 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_40 = torch.conv2d(
            x_h_7,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_7 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_56 = torch.cat((x_id_7, conv2d_38, conv2d_39, conv2d_40), dim=1)
        x_id_7 = conv2d_38 = conv2d_39 = conv2d_40 = None
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_59 = torch._C._nn.gelu(x_58, approximate="none")
        x_58 = None
        x_60 = torch.nn.functional.dropout(x_59, 0.0, False, False)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_7 = l_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_ = (
            None
        )
        x_62 = x_61.mul(reshape_7)
        x_61 = reshape_7 = None
        x_63 = x_62 + x_55
        x_62 = x_55 = None
        split_8 = torch.functional.split(x_63, (240, 48, 48, 48), dim=1)
        x_id_8 = split_8[0]
        x_hw_8 = split_8[1]
        x_w_8 = split_8[2]
        x_h_8 = split_8[3]
        split_8 = None
        conv2d_43 = torch.conv2d(
            x_hw_8,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_8 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_44 = torch.conv2d(
            x_w_8,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_8 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_45 = torch.conv2d(
            x_h_8,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_8 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_64 = torch.cat((x_id_8, conv2d_43, conv2d_44, conv2d_45), dim=1)
        x_id_8 = conv2d_43 = conv2d_44 = conv2d_45 = None
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_67 = torch._C._nn.gelu(x_66, approximate="none")
        x_66 = None
        x_68 = torch.nn.functional.dropout(x_67, 0.0, False, False)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_8 = l_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_ = (
            None
        )
        x_70 = x_69.mul(reshape_8)
        x_69 = reshape_8 = None
        x_71 = x_70 + x_63
        x_70 = x_63 = None
        split_9 = torch.functional.split(x_71, (240, 48, 48, 48), dim=1)
        x_id_9 = split_9[0]
        x_hw_9 = split_9[1]
        x_w_9 = split_9[2]
        x_h_9 = split_9[3]
        split_9 = None
        conv2d_48 = torch.conv2d(
            x_hw_9,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_9 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_49 = torch.conv2d(
            x_w_9,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_9 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_50 = torch.conv2d(
            x_h_9,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_9 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_72 = torch.cat((x_id_9, conv2d_48, conv2d_49, conv2d_50), dim=1)
        x_id_9 = conv2d_48 = conv2d_49 = conv2d_50 = None
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_ = (None)
        x_74 = torch.conv2d(
            x_73,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_73 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_75 = torch._C._nn.gelu(x_74, approximate="none")
        x_74 = None
        x_76 = torch.nn.functional.dropout(x_75, 0.0, False, False)
        x_75 = None
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_76 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_9 = l_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_ = (
            None
        )
        x_78 = x_77.mul(reshape_9)
        x_77 = reshape_9 = None
        x_79 = x_78 + x_71
        x_78 = x_71 = None
        split_10 = torch.functional.split(x_79, (240, 48, 48, 48), dim=1)
        x_id_10 = split_10[0]
        x_hw_10 = split_10[1]
        x_w_10 = split_10[2]
        x_h_10 = split_10[3]
        split_10 = None
        conv2d_53 = torch.conv2d(
            x_hw_10,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_10 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_54 = torch.conv2d(
            x_w_10,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_10 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_55 = torch.conv2d(
            x_h_10,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_10 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_80 = torch.cat((x_id_10, conv2d_53, conv2d_54, conv2d_55), dim=1)
        x_id_10 = conv2d_53 = conv2d_54 = conv2d_55 = None
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_ = (None)
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_83 = torch._C._nn.gelu(x_82, approximate="none")
        x_82 = None
        x_84 = torch.nn.functional.dropout(x_83, 0.0, False, False)
        x_83 = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_84 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_10 = l_self_modules_stages_modules_2_modules_blocks_modules_4_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_parameters_gamma_ = (
            None
        )
        x_86 = x_85.mul(reshape_10)
        x_85 = reshape_10 = None
        x_87 = x_86 + x_79
        x_86 = x_79 = None
        split_11 = torch.functional.split(x_87, (240, 48, 48, 48), dim=1)
        x_id_11 = split_11[0]
        x_hw_11 = split_11[1]
        x_w_11 = split_11[2]
        x_h_11 = split_11[3]
        split_11 = None
        conv2d_58 = torch.conv2d(
            x_hw_11,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_11 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_59 = torch.conv2d(
            x_w_11,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_11 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_60 = torch.conv2d(
            x_h_11,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_11 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_88 = torch.cat((x_id_11, conv2d_58, conv2d_59, conv2d_60), dim=1)
        x_id_11 = conv2d_58 = conv2d_59 = conv2d_60 = None
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_ = (None)
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_91 = torch._C._nn.gelu(x_90, approximate="none")
        x_90 = None
        x_92 = torch.nn.functional.dropout(x_91, 0.0, False, False)
        x_91 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_92 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_11 = l_self_modules_stages_modules_2_modules_blocks_modules_5_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_parameters_gamma_ = (
            None
        )
        x_94 = x_93.mul(reshape_11)
        x_93 = reshape_11 = None
        x_95 = x_94 + x_87
        x_94 = x_87 = None
        split_12 = torch.functional.split(x_95, (240, 48, 48, 48), dim=1)
        x_id_12 = split_12[0]
        x_hw_12 = split_12[1]
        x_w_12 = split_12[2]
        x_h_12 = split_12[3]
        split_12 = None
        conv2d_63 = torch.conv2d(
            x_hw_12,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_12 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_64 = torch.conv2d(
            x_w_12,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_12 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_65 = torch.conv2d(
            x_h_12,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_12 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_96 = torch.cat((x_id_12, conv2d_63, conv2d_64, conv2d_65), dim=1)
        x_id_12 = conv2d_63 = conv2d_64 = conv2d_65 = None
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_96 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_ = (None)
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_99 = torch._C._nn.gelu(x_98, approximate="none")
        x_98 = None
        x_100 = torch.nn.functional.dropout(x_99, 0.0, False, False)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_12 = l_self_modules_stages_modules_2_modules_blocks_modules_6_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_parameters_gamma_ = (
            None
        )
        x_102 = x_101.mul(reshape_12)
        x_101 = reshape_12 = None
        x_103 = x_102 + x_95
        x_102 = x_95 = None
        split_13 = torch.functional.split(x_103, (240, 48, 48, 48), dim=1)
        x_id_13 = split_13[0]
        x_hw_13 = split_13[1]
        x_w_13 = split_13[2]
        x_h_13 = split_13[3]
        split_13 = None
        conv2d_68 = torch.conv2d(
            x_hw_13,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_13 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_69 = torch.conv2d(
            x_w_13,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_13 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_70 = torch.conv2d(
            x_h_13,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_13 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_104 = torch.cat((x_id_13, conv2d_68, conv2d_69, conv2d_70), dim=1)
        x_id_13 = conv2d_68 = conv2d_69 = conv2d_70 = None
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_ = (None)
        x_106 = torch.conv2d(
            x_105,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_105 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_107 = torch._C._nn.gelu(x_106, approximate="none")
        x_106 = None
        x_108 = torch.nn.functional.dropout(x_107, 0.0, False, False)
        x_107 = None
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_108 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_13 = l_self_modules_stages_modules_2_modules_blocks_modules_7_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_parameters_gamma_ = (
            None
        )
        x_110 = x_109.mul(reshape_13)
        x_109 = reshape_13 = None
        x_111 = x_110 + x_103
        x_110 = x_103 = None
        split_14 = torch.functional.split(x_111, (240, 48, 48, 48), dim=1)
        x_id_14 = split_14[0]
        x_hw_14 = split_14[1]
        x_w_14 = split_14[2]
        x_h_14 = split_14[3]
        split_14 = None
        conv2d_73 = torch.conv2d(
            x_hw_14,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_14 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_74 = torch.conv2d(
            x_w_14,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_14 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_75 = torch.conv2d(
            x_h_14,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_14 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_112 = torch.cat((x_id_14, conv2d_73, conv2d_74, conv2d_75), dim=1)
        x_id_14 = conv2d_73 = conv2d_74 = conv2d_75 = None
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_ = (None)
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_113 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_115 = torch._C._nn.gelu(x_114, approximate="none")
        x_114 = None
        x_116 = torch.nn.functional.dropout(x_115, 0.0, False, False)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_116 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_14 = l_self_modules_stages_modules_2_modules_blocks_modules_8_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_parameters_gamma_ = (
            None
        )
        x_118 = x_117.mul(reshape_14)
        x_117 = reshape_14 = None
        x_119 = x_118 + x_111
        x_118 = x_111 = None
        split_15 = torch.functional.split(x_119, (240, 48, 48, 48), dim=1)
        x_id_15 = split_15[0]
        x_hw_15 = split_15[1]
        x_w_15 = split_15[2]
        x_h_15 = split_15[3]
        split_15 = None
        conv2d_78 = torch.conv2d(
            x_hw_15,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_15 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_79 = torch.conv2d(
            x_w_15,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_15 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_80 = torch.conv2d(
            x_h_15,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_15 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_120 = torch.cat((x_id_15, conv2d_78, conv2d_79, conv2d_80), dim=1)
        x_id_15 = conv2d_78 = conv2d_79 = conv2d_80 = None
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_ = (None)
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_123 = torch._C._nn.gelu(x_122, approximate="none")
        x_122 = None
        x_124 = torch.nn.functional.dropout(x_123, 0.0, False, False)
        x_123 = None
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_124 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_15 = l_self_modules_stages_modules_2_modules_blocks_modules_9_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_parameters_gamma_ = (
            None
        )
        x_126 = x_125.mul(reshape_15)
        x_125 = reshape_15 = None
        x_127 = x_126 + x_119
        x_126 = x_119 = None
        split_16 = torch.functional.split(x_127, (240, 48, 48, 48), dim=1)
        x_id_16 = split_16[0]
        x_hw_16 = split_16[1]
        x_w_16 = split_16[2]
        x_h_16 = split_16[3]
        split_16 = None
        conv2d_83 = torch.conv2d(
            x_hw_16,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_16 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_84 = torch.conv2d(
            x_w_16,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_16 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_85 = torch.conv2d(
            x_h_16,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_16 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_128 = torch.cat((x_id_16, conv2d_83, conv2d_84, conv2d_85), dim=1)
        x_id_16 = conv2d_83 = conv2d_84 = conv2d_85 = None
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_ = (None)
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_129 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
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
        reshape_16 = l_self_modules_stages_modules_2_modules_blocks_modules_10_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_parameters_gamma_ = (
            None
        )
        x_134 = x_133.mul(reshape_16)
        x_133 = reshape_16 = None
        x_135 = x_134 + x_127
        x_134 = x_127 = None
        split_17 = torch.functional.split(x_135, (240, 48, 48, 48), dim=1)
        x_id_17 = split_17[0]
        x_hw_17 = split_17[1]
        x_w_17 = split_17[2]
        x_h_17 = split_17[3]
        split_17 = None
        conv2d_88 = torch.conv2d(
            x_hw_17,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_17 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_89 = torch.conv2d(
            x_w_17,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_17 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_90 = torch.conv2d(
            x_h_17,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_17 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_136 = torch.cat((x_id_17, conv2d_88, conv2d_89, conv2d_90), dim=1)
        x_id_17 = conv2d_88 = conv2d_89 = conv2d_90 = None
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_ = (None)
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_139 = torch._C._nn.gelu(x_138, approximate="none")
        x_138 = None
        x_140 = torch.nn.functional.dropout(x_139, 0.0, False, False)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_140 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_17 = l_self_modules_stages_modules_2_modules_blocks_modules_11_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_parameters_gamma_ = (
            None
        )
        x_142 = x_141.mul(reshape_17)
        x_141 = reshape_17 = None
        x_143 = x_142 + x_135
        x_142 = x_135 = None
        split_18 = torch.functional.split(x_143, (240, 48, 48, 48), dim=1)
        x_id_18 = split_18[0]
        x_hw_18 = split_18[1]
        x_w_18 = split_18[2]
        x_h_18 = split_18[3]
        split_18 = None
        conv2d_93 = torch.conv2d(
            x_hw_18,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_18 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_94 = torch.conv2d(
            x_w_18,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_18 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_95 = torch.conv2d(
            x_h_18,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_18 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_144 = torch.cat((x_id_18, conv2d_93, conv2d_94, conv2d_95), dim=1)
        x_id_18 = conv2d_93 = conv2d_94 = conv2d_95 = None
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_144 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_ = (None)
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_145 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_147 = torch._C._nn.gelu(x_146, approximate="none")
        x_146 = None
        x_148 = torch.nn.functional.dropout(x_147, 0.0, False, False)
        x_147 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_18 = l_self_modules_stages_modules_2_modules_blocks_modules_12_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_12_parameters_gamma_ = (
            None
        )
        x_150 = x_149.mul(reshape_18)
        x_149 = reshape_18 = None
        x_151 = x_150 + x_143
        x_150 = x_143 = None
        split_19 = torch.functional.split(x_151, (240, 48, 48, 48), dim=1)
        x_id_19 = split_19[0]
        x_hw_19 = split_19[1]
        x_w_19 = split_19[2]
        x_h_19 = split_19[3]
        split_19 = None
        conv2d_98 = torch.conv2d(
            x_hw_19,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_19 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_99 = torch.conv2d(
            x_w_19,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_19 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_100 = torch.conv2d(
            x_h_19,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_19 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_152 = torch.cat((x_id_19, conv2d_98, conv2d_99, conv2d_100), dim=1)
        x_id_19 = conv2d_98 = conv2d_99 = conv2d_100 = None
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_ = (None)
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_155 = torch._C._nn.gelu(x_154, approximate="none")
        x_154 = None
        x_156 = torch.nn.functional.dropout(x_155, 0.0, False, False)
        x_155 = None
        x_157 = torch.conv2d(
            x_156,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_156 = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_19 = l_self_modules_stages_modules_2_modules_blocks_modules_13_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_13_parameters_gamma_ = (
            None
        )
        x_158 = x_157.mul(reshape_19)
        x_157 = reshape_19 = None
        x_159 = x_158 + x_151
        x_158 = x_151 = None
        split_20 = torch.functional.split(x_159, (240, 48, 48, 48), dim=1)
        x_id_20 = split_20[0]
        x_hw_20 = split_20[1]
        x_w_20 = split_20[2]
        x_h_20 = split_20[3]
        split_20 = None
        conv2d_103 = torch.conv2d(
            x_hw_20,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_20 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_104 = torch.conv2d(
            x_w_20,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_20 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_105 = torch.conv2d(
            x_h_20,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_20 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_160 = torch.cat((x_id_20, conv2d_103, conv2d_104, conv2d_105), dim=1)
        x_id_20 = conv2d_103 = conv2d_104 = conv2d_105 = None
        x_161 = torch.nn.functional.batch_norm(
            x_160,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_160 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_ = (None)
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_163 = torch._C._nn.gelu(x_162, approximate="none")
        x_162 = None
        x_164 = torch.nn.functional.dropout(x_163, 0.0, False, False)
        x_163 = None
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_164 = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_20 = l_self_modules_stages_modules_2_modules_blocks_modules_14_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_14_parameters_gamma_ = (
            None
        )
        x_166 = x_165.mul(reshape_20)
        x_165 = reshape_20 = None
        x_167 = x_166 + x_159
        x_166 = x_159 = None
        split_21 = torch.functional.split(x_167, (240, 48, 48, 48), dim=1)
        x_id_21 = split_21[0]
        x_hw_21 = split_21[1]
        x_w_21 = split_21[2]
        x_h_21 = split_21[3]
        split_21 = None
        conv2d_108 = torch.conv2d(
            x_hw_21,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_21 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_109 = torch.conv2d(
            x_w_21,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_21 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_110 = torch.conv2d(
            x_h_21,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_21 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_168 = torch.cat((x_id_21, conv2d_108, conv2d_109, conv2d_110), dim=1)
        x_id_21 = conv2d_108 = conv2d_109 = conv2d_110 = None
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_168 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_ = (None)
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_171 = torch._C._nn.gelu(x_170, approximate="none")
        x_170 = None
        x_172 = torch.nn.functional.dropout(x_171, 0.0, False, False)
        x_171 = None
        x_173 = torch.conv2d(
            x_172,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_172 = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_21 = l_self_modules_stages_modules_2_modules_blocks_modules_15_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_15_parameters_gamma_ = (
            None
        )
        x_174 = x_173.mul(reshape_21)
        x_173 = reshape_21 = None
        x_175 = x_174 + x_167
        x_174 = x_167 = None
        split_22 = torch.functional.split(x_175, (240, 48, 48, 48), dim=1)
        x_id_22 = split_22[0]
        x_hw_22 = split_22[1]
        x_w_22 = split_22[2]
        x_h_22 = split_22[3]
        split_22 = None
        conv2d_113 = torch.conv2d(
            x_hw_22,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_22 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_114 = torch.conv2d(
            x_w_22,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_22 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_115 = torch.conv2d(
            x_h_22,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_22 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_176 = torch.cat((x_id_22, conv2d_113, conv2d_114, conv2d_115), dim=1)
        x_id_22 = conv2d_113 = conv2d_114 = conv2d_115 = None
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_176 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_ = (None)
        x_178 = torch.conv2d(
            x_177,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_177 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_179 = torch._C._nn.gelu(x_178, approximate="none")
        x_178 = None
        x_180 = torch.nn.functional.dropout(x_179, 0.0, False, False)
        x_179 = None
        x_181 = torch.conv2d(
            x_180,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_180 = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_22 = l_self_modules_stages_modules_2_modules_blocks_modules_16_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_16_parameters_gamma_ = (
            None
        )
        x_182 = x_181.mul(reshape_22)
        x_181 = reshape_22 = None
        x_183 = x_182 + x_175
        x_182 = x_175 = None
        split_23 = torch.functional.split(x_183, (240, 48, 48, 48), dim=1)
        x_id_23 = split_23[0]
        x_hw_23 = split_23[1]
        x_w_23 = split_23[2]
        x_h_23 = split_23[3]
        split_23 = None
        conv2d_118 = torch.conv2d(
            x_hw_23,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_23 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_119 = torch.conv2d(
            x_w_23,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_23 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_120 = torch.conv2d(
            x_h_23,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_23 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_184 = torch.cat((x_id_23, conv2d_118, conv2d_119, conv2d_120), dim=1)
        x_id_23 = conv2d_118 = conv2d_119 = conv2d_120 = None
        x_185 = torch.nn.functional.batch_norm(
            x_184,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_184 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_ = (None)
        x_186 = torch.conv2d(
            x_185,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_185 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_187 = torch._C._nn.gelu(x_186, approximate="none")
        x_186 = None
        x_188 = torch.nn.functional.dropout(x_187, 0.0, False, False)
        x_187 = None
        x_189 = torch.conv2d(
            x_188,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_188 = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_23 = l_self_modules_stages_modules_2_modules_blocks_modules_17_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_17_parameters_gamma_ = (
            None
        )
        x_190 = x_189.mul(reshape_23)
        x_189 = reshape_23 = None
        x_191 = x_190 + x_183
        x_190 = x_183 = None
        split_24 = torch.functional.split(x_191, (240, 48, 48, 48), dim=1)
        x_id_24 = split_24[0]
        x_hw_24 = split_24[1]
        x_w_24 = split_24[2]
        x_h_24 = split_24[3]
        split_24 = None
        conv2d_123 = torch.conv2d(
            x_hw_24,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_24 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_124 = torch.conv2d(
            x_w_24,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_24 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_125 = torch.conv2d(
            x_h_24,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_24 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_192 = torch.cat((x_id_24, conv2d_123, conv2d_124, conv2d_125), dim=1)
        x_id_24 = conv2d_123 = conv2d_124 = conv2d_125 = None
        x_193 = torch.nn.functional.batch_norm(
            x_192,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_192 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_ = (None)
        x_194 = torch.conv2d(
            x_193,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_193 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_195 = torch._C._nn.gelu(x_194, approximate="none")
        x_194 = None
        x_196 = torch.nn.functional.dropout(x_195, 0.0, False, False)
        x_195 = None
        x_197 = torch.conv2d(
            x_196,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_196 = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_24 = l_self_modules_stages_modules_2_modules_blocks_modules_18_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_18_parameters_gamma_ = (
            None
        )
        x_198 = x_197.mul(reshape_24)
        x_197 = reshape_24 = None
        x_199 = x_198 + x_191
        x_198 = x_191 = None
        split_25 = torch.functional.split(x_199, (240, 48, 48, 48), dim=1)
        x_id_25 = split_25[0]
        x_hw_25 = split_25[1]
        x_w_25 = split_25[2]
        x_h_25 = split_25[3]
        split_25 = None
        conv2d_128 = torch.conv2d(
            x_hw_25,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_25 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_129 = torch.conv2d(
            x_w_25,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_25 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_130 = torch.conv2d(
            x_h_25,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_25 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_200 = torch.cat((x_id_25, conv2d_128, conv2d_129, conv2d_130), dim=1)
        x_id_25 = conv2d_128 = conv2d_129 = conv2d_130 = None
        x_201 = torch.nn.functional.batch_norm(
            x_200,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_200 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_ = (None)
        x_202 = torch.conv2d(
            x_201,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_201 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_203 = torch._C._nn.gelu(x_202, approximate="none")
        x_202 = None
        x_204 = torch.nn.functional.dropout(x_203, 0.0, False, False)
        x_203 = None
        x_205 = torch.conv2d(
            x_204,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_204 = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_25 = l_self_modules_stages_modules_2_modules_blocks_modules_19_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_19_parameters_gamma_ = (
            None
        )
        x_206 = x_205.mul(reshape_25)
        x_205 = reshape_25 = None
        x_207 = x_206 + x_199
        x_206 = x_199 = None
        split_26 = torch.functional.split(x_207, (240, 48, 48, 48), dim=1)
        x_id_26 = split_26[0]
        x_hw_26 = split_26[1]
        x_w_26 = split_26[2]
        x_h_26 = split_26[3]
        split_26 = None
        conv2d_133 = torch.conv2d(
            x_hw_26,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_26 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_134 = torch.conv2d(
            x_w_26,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_26 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_135 = torch.conv2d(
            x_h_26,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_26 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_208 = torch.cat((x_id_26, conv2d_133, conv2d_134, conv2d_135), dim=1)
        x_id_26 = conv2d_133 = conv2d_134 = conv2d_135 = None
        x_209 = torch.nn.functional.batch_norm(
            x_208,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_208 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_ = (None)
        x_210 = torch.conv2d(
            x_209,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_209 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_211 = torch._C._nn.gelu(x_210, approximate="none")
        x_210 = None
        x_212 = torch.nn.functional.dropout(x_211, 0.0, False, False)
        x_211 = None
        x_213 = torch.conv2d(
            x_212,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_212 = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_26 = l_self_modules_stages_modules_2_modules_blocks_modules_20_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_20_parameters_gamma_ = (
            None
        )
        x_214 = x_213.mul(reshape_26)
        x_213 = reshape_26 = None
        x_215 = x_214 + x_207
        x_214 = x_207 = None
        split_27 = torch.functional.split(x_215, (240, 48, 48, 48), dim=1)
        x_id_27 = split_27[0]
        x_hw_27 = split_27[1]
        x_w_27 = split_27[2]
        x_h_27 = split_27[3]
        split_27 = None
        conv2d_138 = torch.conv2d(
            x_hw_27,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_27 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_139 = torch.conv2d(
            x_w_27,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_27 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_140 = torch.conv2d(
            x_h_27,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_27 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_216 = torch.cat((x_id_27, conv2d_138, conv2d_139, conv2d_140), dim=1)
        x_id_27 = conv2d_138 = conv2d_139 = conv2d_140 = None
        x_217 = torch.nn.functional.batch_norm(
            x_216,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_216 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_ = (None)
        x_218 = torch.conv2d(
            x_217,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_217 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_219 = torch._C._nn.gelu(x_218, approximate="none")
        x_218 = None
        x_220 = torch.nn.functional.dropout(x_219, 0.0, False, False)
        x_219 = None
        x_221 = torch.conv2d(
            x_220,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_220 = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_27 = l_self_modules_stages_modules_2_modules_blocks_modules_21_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_21_parameters_gamma_ = (
            None
        )
        x_222 = x_221.mul(reshape_27)
        x_221 = reshape_27 = None
        x_223 = x_222 + x_215
        x_222 = x_215 = None
        split_28 = torch.functional.split(x_223, (240, 48, 48, 48), dim=1)
        x_id_28 = split_28[0]
        x_hw_28 = split_28[1]
        x_w_28 = split_28[2]
        x_h_28 = split_28[3]
        split_28 = None
        conv2d_143 = torch.conv2d(
            x_hw_28,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_28 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_144 = torch.conv2d(
            x_w_28,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_28 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_145 = torch.conv2d(
            x_h_28,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_28 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_224 = torch.cat((x_id_28, conv2d_143, conv2d_144, conv2d_145), dim=1)
        x_id_28 = conv2d_143 = conv2d_144 = conv2d_145 = None
        x_225 = torch.nn.functional.batch_norm(
            x_224,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_224 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_ = (None)
        x_226 = torch.conv2d(
            x_225,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_225 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_227 = torch._C._nn.gelu(x_226, approximate="none")
        x_226 = None
        x_228 = torch.nn.functional.dropout(x_227, 0.0, False, False)
        x_227 = None
        x_229 = torch.conv2d(
            x_228,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_228 = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_28 = l_self_modules_stages_modules_2_modules_blocks_modules_22_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_22_parameters_gamma_ = (
            None
        )
        x_230 = x_229.mul(reshape_28)
        x_229 = reshape_28 = None
        x_231 = x_230 + x_223
        x_230 = x_223 = None
        split_29 = torch.functional.split(x_231, (240, 48, 48, 48), dim=1)
        x_id_29 = split_29[0]
        x_hw_29 = split_29[1]
        x_w_29 = split_29[2]
        x_h_29 = split_29[3]
        split_29 = None
        conv2d_148 = torch.conv2d(
            x_hw_29,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_29 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_149 = torch.conv2d(
            x_w_29,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_29 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_150 = torch.conv2d(
            x_h_29,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_29 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_232 = torch.cat((x_id_29, conv2d_148, conv2d_149, conv2d_150), dim=1)
        x_id_29 = conv2d_148 = conv2d_149 = conv2d_150 = None
        x_233 = torch.nn.functional.batch_norm(
            x_232,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_232 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_ = (None)
        x_234 = torch.conv2d(
            x_233,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_233 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_235 = torch._C._nn.gelu(x_234, approximate="none")
        x_234 = None
        x_236 = torch.nn.functional.dropout(x_235, 0.0, False, False)
        x_235 = None
        x_237 = torch.conv2d(
            x_236,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_236 = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_29 = l_self_modules_stages_modules_2_modules_blocks_modules_23_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_23_parameters_gamma_ = (
            None
        )
        x_238 = x_237.mul(reshape_29)
        x_237 = reshape_29 = None
        x_239 = x_238 + x_231
        x_238 = x_231 = None
        split_30 = torch.functional.split(x_239, (240, 48, 48, 48), dim=1)
        x_id_30 = split_30[0]
        x_hw_30 = split_30[1]
        x_w_30 = split_30[2]
        x_h_30 = split_30[3]
        split_30 = None
        conv2d_153 = torch.conv2d(
            x_hw_30,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_30 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_154 = torch.conv2d(
            x_w_30,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_30 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_155 = torch.conv2d(
            x_h_30,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_30 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_240 = torch.cat((x_id_30, conv2d_153, conv2d_154, conv2d_155), dim=1)
        x_id_30 = conv2d_153 = conv2d_154 = conv2d_155 = None
        x_241 = torch.nn.functional.batch_norm(
            x_240,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_240 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_ = (None)
        x_242 = torch.conv2d(
            x_241,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_241 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_243 = torch._C._nn.gelu(x_242, approximate="none")
        x_242 = None
        x_244 = torch.nn.functional.dropout(x_243, 0.0, False, False)
        x_243 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_244 = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_30 = l_self_modules_stages_modules_2_modules_blocks_modules_24_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_24_parameters_gamma_ = (
            None
        )
        x_246 = x_245.mul(reshape_30)
        x_245 = reshape_30 = None
        x_247 = x_246 + x_239
        x_246 = x_239 = None
        split_31 = torch.functional.split(x_247, (240, 48, 48, 48), dim=1)
        x_id_31 = split_31[0]
        x_hw_31 = split_31[1]
        x_w_31 = split_31[2]
        x_h_31 = split_31[3]
        split_31 = None
        conv2d_158 = torch.conv2d(
            x_hw_31,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_31 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_159 = torch.conv2d(
            x_w_31,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_31 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_160 = torch.conv2d(
            x_h_31,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_31 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_248 = torch.cat((x_id_31, conv2d_158, conv2d_159, conv2d_160), dim=1)
        x_id_31 = conv2d_158 = conv2d_159 = conv2d_160 = None
        x_249 = torch.nn.functional.batch_norm(
            x_248,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_248 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_ = (None)
        x_250 = torch.conv2d(
            x_249,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_249 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_251 = torch._C._nn.gelu(x_250, approximate="none")
        x_250 = None
        x_252 = torch.nn.functional.dropout(x_251, 0.0, False, False)
        x_251 = None
        x_253 = torch.conv2d(
            x_252,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_252 = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_31 = l_self_modules_stages_modules_2_modules_blocks_modules_25_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_25_parameters_gamma_ = (
            None
        )
        x_254 = x_253.mul(reshape_31)
        x_253 = reshape_31 = None
        x_255 = x_254 + x_247
        x_254 = x_247 = None
        split_32 = torch.functional.split(x_255, (240, 48, 48, 48), dim=1)
        x_id_32 = split_32[0]
        x_hw_32 = split_32[1]
        x_w_32 = split_32[2]
        x_h_32 = split_32[3]
        split_32 = None
        conv2d_163 = torch.conv2d(
            x_hw_32,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_hw_32 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_164 = torch.conv2d(
            x_w_32,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            48,
        )
        x_w_32 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_165 = torch.conv2d(
            x_h_32,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            48,
        )
        x_h_32 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_256 = torch.cat((x_id_32, conv2d_163, conv2d_164, conv2d_165), dim=1)
        x_id_32 = conv2d_163 = conv2d_164 = conv2d_165 = None
        x_257 = torch.nn.functional.batch_norm(
            x_256,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_256 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_ = (None)
        x_258 = torch.conv2d(
            x_257,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_257 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_259 = torch._C._nn.gelu(x_258, approximate="none")
        x_258 = None
        x_260 = torch.nn.functional.dropout(x_259, 0.0, False, False)
        x_259 = None
        x_261 = torch.conv2d(
            x_260,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_260 = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_32 = l_self_modules_stages_modules_2_modules_blocks_modules_26_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_26_parameters_gamma_ = (
            None
        )
        x_262 = x_261.mul(reshape_32)
        x_261 = reshape_32 = None
        x_263 = x_262 + x_255
        x_262 = x_255 = None
        input_7 = torch.nn.functional.batch_norm(
            x_263,
            l_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_263 = l_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_var_ = l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_ = (None)
        input_8 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_ = (None)
        split_33 = torch.functional.split(input_8, (480, 96, 96, 96), dim=1)
        x_id_33 = split_33[0]
        x_hw_33 = split_33[1]
        x_w_33 = split_33[2]
        x_h_33 = split_33[3]
        split_33 = None
        conv2d_169 = torch.conv2d(
            x_hw_33,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        x_hw_33 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_170 = torch.conv2d(
            x_w_33,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            96,
        )
        x_w_33 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_171 = torch.conv2d(
            x_h_33,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            96,
        )
        x_h_33 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_264 = torch.cat((x_id_33, conv2d_169, conv2d_170, conv2d_171), dim=1)
        x_id_33 = conv2d_169 = conv2d_170 = conv2d_171 = None
        x_265 = torch.nn.functional.batch_norm(
            x_264,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_264 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_266 = torch.conv2d(
            x_265,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_265 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_267 = torch._C._nn.gelu(x_266, approximate="none")
        x_266 = None
        x_268 = torch.nn.functional.dropout(x_267, 0.0, False, False)
        x_267 = None
        x_269 = torch.conv2d(
            x_268,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_268 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_33 = l_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_ = (
            None
        )
        x_270 = x_269.mul(reshape_33)
        x_269 = reshape_33 = None
        x_271 = x_270 + input_8
        x_270 = input_8 = None
        split_34 = torch.functional.split(x_271, (480, 96, 96, 96), dim=1)
        x_id_34 = split_34[0]
        x_hw_34 = split_34[1]
        x_w_34 = split_34[2]
        x_h_34 = split_34[3]
        split_34 = None
        conv2d_174 = torch.conv2d(
            x_hw_34,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        x_hw_34 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_175 = torch.conv2d(
            x_w_34,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            96,
        )
        x_w_34 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_176 = torch.conv2d(
            x_h_34,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            96,
        )
        x_h_34 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_272 = torch.cat((x_id_34, conv2d_174, conv2d_175, conv2d_176), dim=1)
        x_id_34 = conv2d_174 = conv2d_175 = conv2d_176 = None
        x_273 = torch.nn.functional.batch_norm(
            x_272,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_272 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_273 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_275 = torch._C._nn.gelu(x_274, approximate="none")
        x_274 = None
        x_276 = torch.nn.functional.dropout(x_275, 0.0, False, False)
        x_275 = None
        x_277 = torch.conv2d(
            x_276,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_276 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_34 = l_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_ = (
            None
        )
        x_278 = x_277.mul(reshape_34)
        x_277 = reshape_34 = None
        x_279 = x_278 + x_271
        x_278 = x_271 = None
        split_35 = torch.functional.split(x_279, (480, 96, 96, 96), dim=1)
        x_id_35 = split_35[0]
        x_hw_35 = split_35[1]
        x_w_35 = split_35[2]
        x_h_35 = split_35[3]
        split_35 = None
        conv2d_179 = torch.conv2d(
            x_hw_35,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        x_hw_35 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_180 = torch.conv2d(
            x_w_35,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            96,
        )
        x_w_35 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_181 = torch.conv2d(
            x_h_35,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            96,
        )
        x_h_35 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_280 = torch.cat((x_id_35, conv2d_179, conv2d_180, conv2d_181), dim=1)
        x_id_35 = conv2d_179 = conv2d_180 = conv2d_181 = None
        x_281 = torch.nn.functional.batch_norm(
            x_280,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_280 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_282 = torch.conv2d(
            x_281,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_281 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_283 = torch._C._nn.gelu(x_282, approximate="none")
        x_282 = None
        x_284 = torch.nn.functional.dropout(x_283, 0.0, False, False)
        x_283 = None
        x_285 = torch.conv2d(
            x_284,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_284 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_35 = l_self_modules_stages_modules_3_modules_blocks_modules_2_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_parameters_gamma_ = (
            None
        )
        x_286 = x_285.mul(reshape_35)
        x_285 = reshape_35 = None
        x_287 = x_286 + x_279
        x_286 = x_279 = None
        x_288 = torch.nn.functional.adaptive_avg_pool2d(x_287, 1)
        x_287 = None
        x_289 = x_288.flatten(1, -1)
        x_288 = None
        x_290 = torch._C._nn.linear(
            x_289,
            l_self_modules_head_modules_fc1_parameters_weight_,
            l_self_modules_head_modules_fc1_parameters_bias_,
        )
        x_289 = (
            l_self_modules_head_modules_fc1_parameters_weight_
        ) = l_self_modules_head_modules_fc1_parameters_bias_ = None
        x_291 = torch._C._nn.gelu(x_290, approximate="none")
        x_290 = None
        x_292 = torch.nn.functional.layer_norm(
            x_291,
            (2304,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_291 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_293 = torch.nn.functional.dropout(x_292, 0.0, False, False)
        x_292 = None
        x_294 = torch._C._nn.linear(
            x_293,
            l_self_modules_head_modules_fc2_parameters_weight_,
            l_self_modules_head_modules_fc2_parameters_bias_,
        )
        x_293 = (
            l_self_modules_head_modules_fc2_parameters_weight_
        ) = l_self_modules_head_modules_fc2_parameters_bias_ = None
        return (x_294,)
