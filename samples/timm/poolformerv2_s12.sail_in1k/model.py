import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_stem_modules_norm_parameters_weight_ = (
            L_self_modules_stem_modules_norm_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
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
        x_1 = x.permute(0, 2, 3, 1)
        x = None
        x_2 = torch.nn.functional.layer_norm(
            x_1, (64,), l_self_modules_stem_modules_norm_parameters_weight_, None, 1e-06
        )
        x_1 = l_self_modules_stem_modules_norm_parameters_weight_ = None
        x_3 = x_2.permute(0, 3, 1, 2)
        x_2 = None
        sym_sum = torch.sym_sum([-3, s1])
        s1 = None
        floordiv = sym_sum // 4
        sym_sum_1 = torch.sym_sum([1, floordiv])
        floordiv = sym_sum_1 = None
        group_norm = torch.nn.functional.group_norm(
            x_3,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            None
        )
        y = torch._C._nn.avg_pool2d(group_norm, 3, 1, 1, False, False, None)
        sub = y - group_norm
        y = group_norm = None
        x_4 = x_3 + sub
        x_3 = sub = None
        group_norm_1 = torch.nn.functional.group_norm(
            x_4,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            None
        )
        x_5 = torch.conv2d(
            group_norm_1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_1 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu = torch.nn.functional.relu(x_5, inplace=False)
        x_5 = None
        pow_1 = relu**2
        relu = None
        mul = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_1
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_1
        ) = None
        x_6 = (
            mul
            + l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_7 = torch.nn.functional.dropout(x_6, 0.0, False, False)
        x_6 = None
        x_8 = torch.conv2d(
            x_7,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_9 = torch.nn.functional.dropout(x_8, 0.0, False, False)
        x_8 = None
        x_10 = x_4 + x_9
        x_4 = x_9 = None
        group_norm_2 = torch.nn.functional.group_norm(
            x_10,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            None
        )
        y_1 = torch._C._nn.avg_pool2d(group_norm_2, 3, 1, 1, False, False, None)
        sub_1 = y_1 - group_norm_2
        y_1 = group_norm_2 = None
        x_11 = x_10 + sub_1
        x_10 = sub_1 = None
        group_norm_3 = torch.nn.functional.group_norm(
            x_11,
            1,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            None
        )
        x_12 = torch.conv2d(
            group_norm_3,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_3 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_1 = torch.nn.functional.relu(x_12, inplace=False)
        x_12 = None
        pow_2 = relu_1**2
        relu_1 = None
        mul_1 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_2
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_2
        ) = None
        x_13 = (
            mul_1
            + l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_1 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_14 = torch.nn.functional.dropout(x_13, 0.0, False, False)
        x_13 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_16 = torch.nn.functional.dropout(x_15, 0.0, False, False)
        x_15 = None
        x_17 = x_11 + x_16
        x_11 = x_16 = None
        x_18 = x_17.permute(0, 2, 3, 1)
        x_17 = None
        x_19 = torch.nn.functional.layer_norm(
            x_18,
            (64,),
            l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_,
            None,
            1e-06,
        )
        x_18 = l_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_ = (None)
        x_20 = x_19.permute(0, 3, 1, 2)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_ = (None)
        floordiv_1 = sym_sum // 8
        sym_sum_2 = torch.sym_sum([1, floordiv_1])
        floordiv_1 = sym_sum_2 = None
        group_norm_4 = torch.nn.functional.group_norm(
            x_21,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            None
        )
        y_2 = torch._C._nn.avg_pool2d(group_norm_4, 3, 1, 1, False, False, None)
        sub_2 = y_2 - group_norm_4
        y_2 = group_norm_4 = None
        x_22 = x_21 + sub_2
        x_21 = sub_2 = None
        group_norm_5 = torch.nn.functional.group_norm(
            x_22,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            None
        )
        x_23 = torch.conv2d(
            group_norm_5,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_5 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_2 = torch.nn.functional.relu(x_23, inplace=False)
        x_23 = None
        pow_3 = relu_2**2
        relu_2 = None
        mul_2 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_3
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_3
        ) = None
        x_24 = (
            mul_2
            + l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul_2 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_25 = torch.nn.functional.dropout(x_24, 0.0, False, False)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_27 = torch.nn.functional.dropout(x_26, 0.0, False, False)
        x_26 = None
        x_28 = x_22 + x_27
        x_22 = x_27 = None
        group_norm_6 = torch.nn.functional.group_norm(
            x_28,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            None
        )
        y_3 = torch._C._nn.avg_pool2d(group_norm_6, 3, 1, 1, False, False, None)
        sub_3 = y_3 - group_norm_6
        y_3 = group_norm_6 = None
        x_29 = x_28 + sub_3
        x_28 = sub_3 = None
        group_norm_7 = torch.nn.functional.group_norm(
            x_29,
            1,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            None
        )
        x_30 = torch.conv2d(
            group_norm_7,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_7 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_3 = torch.nn.functional.relu(x_30, inplace=False)
        x_30 = None
        pow_4 = relu_3**2
        relu_3 = None
        mul_3 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_4
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_4
        ) = None
        x_31 = (
            mul_3
            + l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_3 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_32 = torch.nn.functional.dropout(x_31, 0.0, False, False)
        x_31 = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_34 = torch.nn.functional.dropout(x_33, 0.0, False, False)
        x_33 = None
        x_35 = x_29 + x_34
        x_29 = x_34 = None
        x_36 = x_35.permute(0, 2, 3, 1)
        x_35 = None
        x_37 = torch.nn.functional.layer_norm(
            x_36,
            (128,),
            l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_,
            None,
            1e-06,
        )
        x_36 = l_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_ = (None)
        x_38 = x_37.permute(0, 3, 1, 2)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_ = (None)
        floordiv_2 = sym_sum // 16
        sym_sum_3 = torch.sym_sum([1, floordiv_2])
        floordiv_2 = sym_sum_3 = None
        view = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_4 = x_39 * view
        view = None
        group_norm_8 = torch.nn.functional.group_norm(
            x_39,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_39 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (None)
        y_4 = torch._C._nn.avg_pool2d(group_norm_8, 3, 1, 1, False, False, None)
        sub_4 = y_4 - group_norm_8
        y_4 = group_norm_8 = None
        x_40 = mul_4 + sub_4
        mul_4 = sub_4 = None
        view_1 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_5 = x_40 * view_1
        view_1 = None
        group_norm_9 = torch.nn.functional.group_norm(
            x_40,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_40 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (None)
        x_41 = torch.conv2d(
            group_norm_9,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_9 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_4 = torch.nn.functional.relu(x_41, inplace=False)
        x_41 = None
        pow_5 = relu_4**2
        relu_4 = None
        mul_6 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_5
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_5
        ) = None
        x_42 = (
            mul_6
            + l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul_6 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_43 = torch.nn.functional.dropout(x_42, 0.0, False, False)
        x_42 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_45 = torch.nn.functional.dropout(x_44, 0.0, False, False)
        x_44 = None
        x_46 = mul_5 + x_45
        mul_5 = x_45 = None
        view_2 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_7 = x_46 * view_2
        view_2 = None
        group_norm_10 = torch.nn.functional.group_norm(
            x_46,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_46 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (None)
        y_5 = torch._C._nn.avg_pool2d(group_norm_10, 3, 1, 1, False, False, None)
        sub_5 = y_5 - group_norm_10
        y_5 = group_norm_10 = None
        x_47 = mul_7 + sub_5
        mul_7 = sub_5 = None
        view_3 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_8 = x_47 * view_3
        view_3 = None
        group_norm_11 = torch.nn.functional.group_norm(
            x_47,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_47 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (None)
        x_48 = torch.conv2d(
            group_norm_11,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_11 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_5 = torch.nn.functional.relu(x_48, inplace=False)
        x_48 = None
        pow_6 = relu_5**2
        relu_5 = None
        mul_9 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_6
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_6
        ) = None
        x_49 = (
            mul_9
            + l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_9 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_50 = torch.nn.functional.dropout(x_49, 0.0, False, False)
        x_49 = None
        x_51 = torch.conv2d(
            x_50,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_50 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_52 = torch.nn.functional.dropout(x_51, 0.0, False, False)
        x_51 = None
        x_53 = mul_8 + x_52
        mul_8 = x_52 = None
        view_4 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_10 = x_53 * view_4
        view_4 = None
        group_norm_12 = torch.nn.functional.group_norm(
            x_53,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_53 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (None)
        y_6 = torch._C._nn.avg_pool2d(group_norm_12, 3, 1, 1, False, False, None)
        sub_6 = y_6 - group_norm_12
        y_6 = group_norm_12 = None
        x_54 = mul_10 + sub_6
        mul_10 = sub_6 = None
        view_5 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_11 = x_54 * view_5
        view_5 = None
        group_norm_13 = torch.nn.functional.group_norm(
            x_54,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_54 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (None)
        x_55 = torch.conv2d(
            group_norm_13,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_13 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_6 = torch.nn.functional.relu(x_55, inplace=False)
        x_55 = None
        pow_7 = relu_6**2
        relu_6 = None
        mul_12 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_
            * pow_7
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_scale_ = (
            pow_7
        ) = None
        x_56 = (
            mul_12
            + l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_
        )
        mul_12 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_act_parameters_bias_ = (None)
        x_57 = torch.nn.functional.dropout(x_56, 0.0, False, False)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_59 = torch.nn.functional.dropout(x_58, 0.0, False, False)
        x_58 = None
        x_60 = mul_11 + x_59
        mul_11 = x_59 = None
        view_6 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_13 = x_60 * view_6
        view_6 = None
        group_norm_14 = torch.nn.functional.group_norm(
            x_60,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_60 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (None)
        y_7 = torch._C._nn.avg_pool2d(group_norm_14, 3, 1, 1, False, False, None)
        sub_7 = y_7 - group_norm_14
        y_7 = group_norm_14 = None
        x_61 = mul_13 + sub_7
        mul_13 = sub_7 = None
        view_7 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_14 = x_61 * view_7
        view_7 = None
        group_norm_15 = torch.nn.functional.group_norm(
            x_61,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_61 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (None)
        x_62 = torch.conv2d(
            group_norm_15,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_15 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_7 = torch.nn.functional.relu(x_62, inplace=False)
        x_62 = None
        pow_8 = relu_7**2
        relu_7 = None
        mul_15 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_
            * pow_8
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_scale_ = (
            pow_8
        ) = None
        x_63 = (
            mul_15
            + l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_
        )
        mul_15 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_act_parameters_bias_ = (None)
        x_64 = torch.nn.functional.dropout(x_63, 0.0, False, False)
        x_63 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_66 = torch.nn.functional.dropout(x_65, 0.0, False, False)
        x_65 = None
        x_67 = mul_14 + x_66
        mul_14 = x_66 = None
        view_8 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_16 = x_67 * view_8
        view_8 = None
        group_norm_16 = torch.nn.functional.group_norm(
            x_67,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_67 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (None)
        y_8 = torch._C._nn.avg_pool2d(group_norm_16, 3, 1, 1, False, False, None)
        sub_8 = y_8 - group_norm_16
        y_8 = group_norm_16 = None
        x_68 = mul_16 + sub_8
        mul_16 = sub_8 = None
        view_9 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_17 = x_68 * view_9
        view_9 = None
        group_norm_17 = torch.nn.functional.group_norm(
            x_68,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_68 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (None)
        x_69 = torch.conv2d(
            group_norm_17,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_17 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_8 = torch.nn.functional.relu(x_69, inplace=False)
        x_69 = None
        pow_9 = relu_8**2
        relu_8 = None
        mul_18 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_
            * pow_9
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_scale_ = (
            pow_9
        ) = None
        x_70 = (
            mul_18
            + l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_
        )
        mul_18 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_act_parameters_bias_ = (None)
        x_71 = torch.nn.functional.dropout(x_70, 0.0, False, False)
        x_70 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_71 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_73 = torch.nn.functional.dropout(x_72, 0.0, False, False)
        x_72 = None
        x_74 = mul_17 + x_73
        mul_17 = x_73 = None
        view_10 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_19 = x_74 * view_10
        view_10 = None
        group_norm_18 = torch.nn.functional.group_norm(
            x_74,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_74 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (None)
        y_9 = torch._C._nn.avg_pool2d(group_norm_18, 3, 1, 1, False, False, None)
        sub_9 = y_9 - group_norm_18
        y_9 = group_norm_18 = None
        x_75 = mul_19 + sub_9
        mul_19 = sub_9 = None
        view_11 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_.view(
            (320, 1, 1)
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_20 = x_75 * view_11
        view_11 = None
        group_norm_19 = torch.nn.functional.group_norm(
            x_75,
            1,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_75 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (None)
        x_76 = torch.conv2d(
            group_norm_19,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_19 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_9 = torch.nn.functional.relu(x_76, inplace=False)
        x_76 = None
        pow_10 = relu_9**2
        relu_9 = None
        mul_21 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_
            * pow_10
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_scale_ = (
            pow_10
        ) = None
        x_77 = (
            mul_21
            + l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_
        )
        mul_21 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_act_parameters_bias_ = (None)
        x_78 = torch.nn.functional.dropout(x_77, 0.0, False, False)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_80 = torch.nn.functional.dropout(x_79, 0.0, False, False)
        x_79 = None
        x_81 = mul_20 + x_80
        mul_20 = x_80 = None
        x_82 = x_81.permute(0, 2, 3, 1)
        x_81 = None
        x_83 = torch.nn.functional.layer_norm(
            x_82,
            (320,),
            l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_,
            None,
            1e-06,
        )
        x_82 = l_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_ = (None)
        x_84 = x_83.permute(0, 3, 1, 2)
        x_83 = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_84 = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_ = (None)
        floordiv_3 = sym_sum // 32
        sym_sum = None
        sym_sum_4 = torch.sym_sum([1, floordiv_3])
        floordiv_3 = sym_sum_4 = None
        view_12 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_22 = x_85 * view_12
        view_12 = None
        group_norm_20 = torch.nn.functional.group_norm(
            x_85,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_85 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (None)
        y_10 = torch._C._nn.avg_pool2d(group_norm_20, 3, 1, 1, False, False, None)
        sub_10 = y_10 - group_norm_20
        y_10 = group_norm_20 = None
        x_86 = mul_22 + sub_10
        mul_22 = sub_10 = None
        view_13 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_23 = x_86 * view_13
        view_13 = None
        group_norm_21 = torch.nn.functional.group_norm(
            x_86,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_86 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (None)
        x_87 = torch.conv2d(
            group_norm_21,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_21 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_10 = torch.nn.functional.relu(x_87, inplace=False)
        x_87 = None
        pow_11 = relu_10**2
        relu_10 = None
        mul_24 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_
            * pow_11
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_scale_ = (
            pow_11
        ) = None
        x_88 = (
            mul_24
            + l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_
        )
        mul_24 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_act_parameters_bias_ = (None)
        x_89 = torch.nn.functional.dropout(x_88, 0.0, False, False)
        x_88 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        x_92 = mul_23 + x_91
        mul_23 = x_91 = None
        view_14 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale1_parameters_scale_ = (
            None
        )
        mul_25 = x_92 * view_14
        view_14 = None
        group_norm_22 = torch.nn.functional.group_norm(
            x_92,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            None,
            1e-06,
        )
        x_92 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (None)
        y_11 = torch._C._nn.avg_pool2d(group_norm_22, 3, 1, 1, False, False, None)
        sub_11 = y_11 - group_norm_22
        y_11 = group_norm_22 = None
        x_93 = mul_25 + sub_11
        mul_25 = sub_11 = None
        view_15 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_.view(
            (512, 1, 1)
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_res_scale2_parameters_scale_ = (
            None
        )
        mul_26 = x_93 * view_15
        view_15 = None
        group_norm_23 = torch.nn.functional.group_norm(
            x_93,
            1,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            None,
            1e-06,
        )
        x_93 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (None)
        x_94 = torch.conv2d(
            group_norm_23,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_23 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (None)
        relu_11 = torch.nn.functional.relu(x_94, inplace=False)
        x_94 = None
        pow_12 = relu_11**2
        relu_11 = None
        mul_27 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_
            * pow_12
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_scale_ = (
            pow_12
        ) = None
        x_95 = (
            mul_27
            + l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_
        )
        mul_27 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_act_parameters_bias_ = (None)
        x_96 = torch.nn.functional.dropout(x_95, 0.0, False, False)
        x_95 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (None)
        x_98 = torch.nn.functional.dropout(x_97, 0.0, False, False)
        x_97 = None
        x_99 = mul_26 + x_98
        mul_26 = x_98 = None
        x_100 = torch.nn.functional.adaptive_avg_pool2d(x_99, 1)
        x_99 = None
        x_101 = x_100.permute(0, 2, 3, 1)
        x_100 = None
        x_102 = torch.nn.functional.layer_norm(
            x_101,
            (512,),
            l_self_modules_head_modules_norm_parameters_weight_,
            l_self_modules_head_modules_norm_parameters_bias_,
            1e-06,
        )
        x_101 = (
            l_self_modules_head_modules_norm_parameters_weight_
        ) = l_self_modules_head_modules_norm_parameters_bias_ = None
        x_103 = x_102.permute(0, 3, 1, 2)
        x_102 = None
        x_104 = x_103.flatten(1, -1)
        x_103 = None
        x_105 = torch._C._nn.linear(
            x_104,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_104 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_105,)
