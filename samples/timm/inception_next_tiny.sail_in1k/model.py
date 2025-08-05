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
        input_7 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = l_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_var_ = l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_ = (None)
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
        split_15 = torch.functional.split(input_8, (480, 96, 96, 96), dim=1)
        x_id_15 = split_15[0]
        x_hw_15 = split_15[1]
        x_w_15 = split_15[2]
        x_h_15 = split_15[3]
        split_15 = None
        conv2d_79 = torch.conv2d(
            x_hw_15,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        x_hw_15 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_80 = torch.conv2d(
            x_w_15,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            96,
        )
        x_w_15 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_81 = torch.conv2d(
            x_h_15,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            96,
        )
        x_h_15 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_120 = torch.cat((x_id_15, conv2d_79, conv2d_80, conv2d_81), dim=1)
        x_id_15 = conv2d_79 = conv2d_80 = conv2d_81 = None
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_ = (None)
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_123 = torch._C._nn.gelu(x_122, approximate="none")
        x_122 = None
        x_124 = torch.nn.functional.dropout(x_123, 0.0, False, False)
        x_123 = None
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_124 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_15 = l_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_ = (
            None
        )
        x_126 = x_125.mul(reshape_15)
        x_125 = reshape_15 = None
        x_127 = x_126 + input_8
        x_126 = input_8 = None
        split_16 = torch.functional.split(x_127, (480, 96, 96, 96), dim=1)
        x_id_16 = split_16[0]
        x_hw_16 = split_16[1]
        x_w_16 = split_16[2]
        x_h_16 = split_16[3]
        split_16 = None
        conv2d_84 = torch.conv2d(
            x_hw_16,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        x_hw_16 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_85 = torch.conv2d(
            x_w_16,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            96,
        )
        x_w_16 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_86 = torch.conv2d(
            x_h_16,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            96,
        )
        x_h_16 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_128 = torch.cat((x_id_16, conv2d_84, conv2d_85, conv2d_86), dim=1)
        x_id_16 = conv2d_84 = conv2d_85 = conv2d_86 = None
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_ = (None)
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_129 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_131 = torch._C._nn.gelu(x_130, approximate="none")
        x_130 = None
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_132 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_16 = l_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_ = (
            None
        )
        x_134 = x_133.mul(reshape_16)
        x_133 = reshape_16 = None
        x_135 = x_134 + x_127
        x_134 = x_127 = None
        split_17 = torch.functional.split(x_135, (480, 96, 96, 96), dim=1)
        x_id_17 = split_17[0]
        x_hw_17 = split_17[1]
        x_w_17 = split_17[2]
        x_h_17 = split_17[3]
        split_17 = None
        conv2d_89 = torch.conv2d(
            x_hw_17,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        x_hw_17 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_ = (None)
        conv2d_90 = torch.conv2d(
            x_w_17,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_,
            (1, 1),
            (0, 5),
            (1, 1),
            96,
        )
        x_w_17 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_ = (None)
        conv2d_91 = torch.conv2d(
            x_h_17,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_,
            (1, 1),
            (5, 0),
            (1, 1),
            96,
        )
        x_h_17 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_ = (None)
        x_136 = torch.cat((x_id_17, conv2d_89, conv2d_90, conv2d_91), dim=1)
        x_id_17 = conv2d_89 = conv2d_90 = conv2d_91 = None
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_ = (None)
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_139 = torch._C._nn.gelu(x_138, approximate="none")
        x_138 = None
        x_140 = torch.nn.functional.dropout(x_139, 0.0, False, False)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_140 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        reshape_17 = l_self_modules_stages_modules_3_modules_blocks_modules_2_parameters_gamma_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_parameters_gamma_ = (
            None
        )
        x_142 = x_141.mul(reshape_17)
        x_141 = reshape_17 = None
        x_143 = x_142 + x_135
        x_142 = x_135 = None
        x_144 = torch.nn.functional.adaptive_avg_pool2d(x_143, 1)
        x_143 = None
        x_145 = x_144.flatten(1, -1)
        x_144 = None
        x_146 = torch._C._nn.linear(
            x_145,
            l_self_modules_head_modules_fc1_parameters_weight_,
            l_self_modules_head_modules_fc1_parameters_bias_,
        )
        x_145 = (
            l_self_modules_head_modules_fc1_parameters_weight_
        ) = l_self_modules_head_modules_fc1_parameters_bias_ = None
        x_147 = torch._C._nn.gelu(x_146, approximate="none")
        x_146 = None
        x_148 = torch.nn.functional.layer_norm(
            x_147,
            (2304,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_147 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_149 = torch.nn.functional.dropout(x_148, 0.0, False, False)
        x_148 = None
        x_150 = torch._C._nn.linear(
            x_149,
            l_self_modules_head_modules_fc2_parameters_weight_,
            l_self_modules_head_modules_fc2_parameters_bias_,
        )
        x_149 = (
            l_self_modules_head_modules_fc2_parameters_weight_
        ) = l_self_modules_head_modules_fc2_parameters_bias_ = None
        return (x_150,)
