import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_4_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_4_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_4_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_4_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_5_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_5_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_5_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_5_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_6_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_6_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_6_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_6_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_7_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_7_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_7_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_7_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_20_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_20_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_21_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_21_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_22_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_22_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_23_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_23_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv_1x1_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv_1x1_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv_1x1_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv_1x1_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv_1x1_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv_1x1_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_conv_kxk_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv_kxk_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv_kxk_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv_kxk_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv_kxk_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv_kxk_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_attn_modules_fc1_parameters_weight_ = (
            L_self_modules_stem_modules_attn_modules_fc1_parameters_weight_
        )
        l_self_modules_stem_modules_attn_modules_fc1_parameters_bias_ = (
            L_self_modules_stem_modules_attn_modules_fc1_parameters_bias_
        )
        l_self_modules_stem_modules_attn_modules_fc2_parameters_weight_ = (
            L_self_modules_stem_modules_attn_modules_fc2_parameters_weight_
        )
        l_self_modules_stem_modules_attn_modules_fc2_parameters_bias_ = (
            L_self_modules_stem_modules_attn_modules_fc2_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_identity_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_identity_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_1_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_identity_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_identity_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_2_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_0_modules_3_modules_identity_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_identity_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_3_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_4_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_4_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_4_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_0_modules_4_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_0_modules_4_modules_identity_parameters_weight_ = L_self_modules_stages_modules_0_modules_4_modules_identity_parameters_weight_
        l_self_modules_stages_modules_0_modules_4_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_4_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_5_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_5_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_5_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_0_modules_5_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_0_modules_5_modules_identity_parameters_weight_ = L_self_modules_stages_modules_0_modules_5_modules_identity_parameters_weight_
        l_self_modules_stages_modules_0_modules_5_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_5_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_6_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_6_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_6_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_0_modules_6_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_0_modules_6_modules_identity_parameters_weight_ = L_self_modules_stages_modules_0_modules_6_modules_identity_parameters_weight_
        l_self_modules_stages_modules_0_modules_6_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_6_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_7_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_7_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_7_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_0_modules_7_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_0_modules_7_modules_identity_parameters_weight_ = L_self_modules_stages_modules_0_modules_7_modules_identity_parameters_weight_
        l_self_modules_stages_modules_0_modules_7_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_7_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_1_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_2_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_3_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_3_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_4_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_5_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_5_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_6_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_6_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_7_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_7_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_8_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_8_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_9_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_9_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_10_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_10_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_11_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_11_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_12_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_12_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_13_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_13_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_1_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_2_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_3_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_4_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_5_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_6_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_7_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_7_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_8_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_9_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_9_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_10_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_11_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_11_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_12_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_13_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_13_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_14_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_15_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_15_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_16_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_16_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_17_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_17_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_18_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_18_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_19_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_19_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_20_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_20_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_20_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_20_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_20_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_20_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_21_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_21_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_21_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_21_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_21_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_21_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_22_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_22_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_22_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_22_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_22_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_22_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_23_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_23_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_23_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_23_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_23_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_23_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_bias_
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stem_modules_conv_1x1_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv_1x1_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv_1x1_modules_bn_parameters_bias_ = None
        x_2 = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_stem_modules_conv_kxk_modules_conv_parameters_weight_
        ) = None
        x_3 = torch.nn.functional.batch_norm(
            x_2,
            l_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_2 = (
            l_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv_kxk_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv_kxk_modules_bn_parameters_bias_ = None
        x_4 = x_1 + x_3
        x_1 = x_3 = None
        x_se = x_4.mean((2, 3), keepdim=True)
        x_se_1 = torch.conv2d(
            x_se,
            l_self_modules_stem_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stem_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se = (
            l_self_modules_stem_modules_attn_modules_fc1_parameters_weight_
        ) = l_self_modules_stem_modules_attn_modules_fc1_parameters_bias_ = None
        x_se_2 = torch.nn.functional.relu(x_se_1, inplace=True)
        x_se_1 = None
        x_se_3 = torch.conv2d(
            x_se_2,
            l_self_modules_stem_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stem_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_2 = (
            l_self_modules_stem_modules_attn_modules_fc2_parameters_weight_
        ) = l_self_modules_stem_modules_attn_modules_fc2_parameters_bias_ = None
        sigmoid = x_se_3.sigmoid()
        x_se_3 = None
        x_5 = x_4 * sigmoid
        x_4 = sigmoid = None
        x_6 = torch.nn.functional.relu(x_5, inplace=True)
        x_5 = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_9 = torch.conv2d(
            x_6,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_11 = x_8 + x_10
        x_8 = x_10 = None
        x_se_4 = x_11.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_6 = torch.nn.functional.relu(x_se_5, inplace=True)
        x_se_5 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_1 = x_se_7.sigmoid()
        x_se_7 = None
        x_12 = x_11 * sigmoid_1
        x_11 = sigmoid_1 = None
        input_1 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        x_13 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_0_modules_1_modules_identity_parameters_bias_
        ) = None
        x_14 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_16 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_18 = x_15 + x_17
        x_15 = x_17 = None
        x_18 += x_13
        x_19 = x_18
        x_18 = x_13 = None
        x_se_8 = x_19.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_10 = torch.nn.functional.relu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_2 = x_se_11.sigmoid()
        x_se_11 = None
        x_20 = x_19 * sigmoid_2
        x_19 = sigmoid_2 = None
        input_2 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        x_21 = torch.nn.functional.batch_norm(
            input_2,
            l_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_0_modules_2_modules_identity_parameters_bias_
        ) = None
        x_22 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_24 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_2 = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_26 = x_23 + x_25
        x_23 = x_25 = None
        x_26 += x_21
        x_27 = x_26
        x_26 = x_21 = None
        x_se_12 = x_27.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = l_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_14 = torch.nn.functional.relu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = l_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_3 = x_se_15.sigmoid()
        x_se_15 = None
        x_28 = x_27 * sigmoid_3
        x_27 = sigmoid_3 = None
        input_3 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        x_29 = torch.nn.functional.batch_norm(
            input_3,
            l_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_3_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_0_modules_3_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_0_modules_3_modules_identity_parameters_bias_
        ) = None
        x_30 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_32 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_34 = x_31 + x_33
        x_31 = x_33 = None
        x_34 += x_29
        x_35 = x_34
        x_34 = x_29 = None
        x_se_16 = x_35.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = l_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_18 = torch.nn.functional.relu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = l_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_3_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_4 = x_se_19.sigmoid()
        x_se_19 = None
        x_36 = x_35 * sigmoid_4
        x_35 = sigmoid_4 = None
        input_4 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        x_37 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_stages_modules_0_modules_4_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_4_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_4_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_0_modules_4_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_4_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_4_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_0_modules_4_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_0_modules_4_modules_identity_parameters_bias_
        ) = None
        x_38 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_4_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_40 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_4 = l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_4_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_42 = x_39 + x_41
        x_39 = x_41 = None
        x_42 += x_37
        x_43 = x_42
        x_42 = x_37 = None
        x_se_20 = x_43.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = l_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_22 = torch.nn.functional.relu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = l_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_4_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_5 = x_se_23.sigmoid()
        x_se_23 = None
        x_44 = x_43 * sigmoid_5
        x_43 = sigmoid_5 = None
        input_5 = torch.nn.functional.relu(x_44, inplace=True)
        x_44 = None
        x_45 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_stages_modules_0_modules_5_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_5_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_5_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_0_modules_5_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_5_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_5_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_0_modules_5_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_0_modules_5_modules_identity_parameters_bias_
        ) = None
        x_46 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_5_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_48 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_5_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_50 = x_47 + x_49
        x_47 = x_49 = None
        x_50 += x_45
        x_51 = x_50
        x_50 = x_45 = None
        x_se_24 = x_51.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = l_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_26 = torch.nn.functional.relu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = l_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_5_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_6 = x_se_27.sigmoid()
        x_se_27 = None
        x_52 = x_51 * sigmoid_6
        x_51 = sigmoid_6 = None
        input_6 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        x_53 = torch.nn.functional.batch_norm(
            input_6,
            l_self_modules_stages_modules_0_modules_6_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_6_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_6_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_0_modules_6_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_6_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_6_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_0_modules_6_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_0_modules_6_modules_identity_parameters_bias_
        ) = None
        x_54 = torch.conv2d(
            input_6,
            l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_6_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_56 = torch.conv2d(
            input_6,
            l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_6 = l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_6_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_58 = x_55 + x_57
        x_55 = x_57 = None
        x_58 += x_53
        x_59 = x_58
        x_58 = x_53 = None
        x_se_28 = x_59.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = l_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_30 = torch.nn.functional.relu(x_se_29, inplace=True)
        x_se_29 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = l_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_6_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_7 = x_se_31.sigmoid()
        x_se_31 = None
        x_60 = x_59 * sigmoid_7
        x_59 = sigmoid_7 = None
        input_7 = torch.nn.functional.relu(x_60, inplace=True)
        x_60 = None
        x_61 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_stages_modules_0_modules_7_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_7_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_7_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_0_modules_7_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_7_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_7_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_0_modules_7_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_0_modules_7_modules_identity_parameters_bias_
        ) = None
        x_62 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_7_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_64 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_7_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_66 = x_63 + x_65
        x_63 = x_65 = None
        x_66 += x_61
        x_67 = x_66
        x_66 = x_61 = None
        x_se_32 = x_67.mean((2, 3), keepdim=True)
        x_se_33 = torch.conv2d(
            x_se_32,
            l_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_32 = l_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_34 = torch.nn.functional.relu(x_se_33, inplace=True)
        x_se_33 = None
        x_se_35 = torch.conv2d(
            x_se_34,
            l_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_34 = l_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_7_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_8 = x_se_35.sigmoid()
        x_se_35 = None
        x_68 = x_67 * sigmoid_8
        x_67 = sigmoid_8 = None
        input_8 = torch.nn.functional.relu(x_68, inplace=True)
        x_68 = None
        x_69 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_71 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_8 = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_71 = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_73 = x_70 + x_72
        x_70 = x_72 = None
        x_se_36 = x_73.mean((2, 3), keepdim=True)
        x_se_37 = torch.conv2d(
            x_se_36,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_36 = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_38 = torch.nn.functional.relu(x_se_37, inplace=True)
        x_se_37 = None
        x_se_39 = torch.conv2d(
            x_se_38,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_38 = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_9 = x_se_39.sigmoid()
        x_se_39 = None
        x_74 = x_73 * sigmoid_9
        x_73 = sigmoid_9 = None
        input_9 = torch.nn.functional.relu(x_74, inplace=True)
        x_74 = None
        x_75 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_1_modules_identity_parameters_bias_
        ) = None
        x_76 = torch.conv2d(
            input_9,
            l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_78 = torch.conv2d(
            input_9,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_9 = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_78 = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_80 = x_77 + x_79
        x_77 = x_79 = None
        x_80 += x_75
        x_81 = x_80
        x_80 = x_75 = None
        x_se_40 = x_81.mean((2, 3), keepdim=True)
        x_se_41 = torch.conv2d(
            x_se_40,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_40 = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_42 = torch.nn.functional.relu(x_se_41, inplace=True)
        x_se_41 = None
        x_se_43 = torch.conv2d(
            x_se_42,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_42 = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_10 = x_se_43.sigmoid()
        x_se_43 = None
        x_82 = x_81 * sigmoid_10
        x_81 = sigmoid_10 = None
        input_10 = torch.nn.functional.relu(x_82, inplace=True)
        x_82 = None
        x_83 = torch.nn.functional.batch_norm(
            input_10,
            l_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_2_modules_identity_parameters_bias_
        ) = None
        x_84 = torch.conv2d(
            input_10,
            l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_86 = torch.conv2d(
            input_10,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_10 = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_88 = x_85 + x_87
        x_85 = x_87 = None
        x_88 += x_83
        x_89 = x_88
        x_88 = x_83 = None
        x_se_44 = x_89.mean((2, 3), keepdim=True)
        x_se_45 = torch.conv2d(
            x_se_44,
            l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_44 = l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_46 = torch.nn.functional.relu(x_se_45, inplace=True)
        x_se_45 = None
        x_se_47 = torch.conv2d(
            x_se_46,
            l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_46 = l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_11 = x_se_47.sigmoid()
        x_se_47 = None
        x_90 = x_89 * sigmoid_11
        x_89 = sigmoid_11 = None
        input_11 = torch.nn.functional.relu(x_90, inplace=True)
        x_90 = None
        x_91 = torch.nn.functional.batch_norm(
            input_11,
            l_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_3_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_3_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_3_modules_identity_parameters_bias_
        ) = None
        x_92 = torch.conv2d(
            input_11,
            l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_94 = torch.conv2d(
            input_11,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_11 = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_96 = x_93 + x_95
        x_93 = x_95 = None
        x_96 += x_91
        x_97 = x_96
        x_96 = x_91 = None
        x_se_48 = x_97.mean((2, 3), keepdim=True)
        x_se_49 = torch.conv2d(
            x_se_48,
            l_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_48 = l_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_50 = torch.nn.functional.relu(x_se_49, inplace=True)
        x_se_49 = None
        x_se_51 = torch.conv2d(
            x_se_50,
            l_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_50 = l_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_12 = x_se_51.sigmoid()
        x_se_51 = None
        x_98 = x_97 * sigmoid_12
        x_97 = sigmoid_12 = None
        input_12 = torch.nn.functional.relu(x_98, inplace=True)
        x_98 = None
        x_99 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_4_modules_identity_parameters_bias_
        ) = None
        x_100 = torch.conv2d(
            input_12,
            l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_102 = torch.conv2d(
            input_12,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_12 = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_104 = x_101 + x_103
        x_101 = x_103 = None
        x_104 += x_99
        x_105 = x_104
        x_104 = x_99 = None
        x_se_52 = x_105.mean((2, 3), keepdim=True)
        x_se_53 = torch.conv2d(
            x_se_52,
            l_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_52 = l_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_54 = torch.nn.functional.relu(x_se_53, inplace=True)
        x_se_53 = None
        x_se_55 = torch.conv2d(
            x_se_54,
            l_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_54 = l_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_13 = x_se_55.sigmoid()
        x_se_55 = None
        x_106 = x_105 * sigmoid_13
        x_105 = sigmoid_13 = None
        input_13 = torch.nn.functional.relu(x_106, inplace=True)
        x_106 = None
        x_107 = torch.nn.functional.batch_norm(
            input_13,
            l_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_5_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_5_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_5_modules_identity_parameters_bias_
        ) = None
        x_108 = torch.conv2d(
            input_13,
            l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_110 = torch.conv2d(
            input_13,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_13 = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_112 = x_109 + x_111
        x_109 = x_111 = None
        x_112 += x_107
        x_113 = x_112
        x_112 = x_107 = None
        x_se_56 = x_113.mean((2, 3), keepdim=True)
        x_se_57 = torch.conv2d(
            x_se_56,
            l_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_56 = l_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_58 = torch.nn.functional.relu(x_se_57, inplace=True)
        x_se_57 = None
        x_se_59 = torch.conv2d(
            x_se_58,
            l_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_58 = l_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_14 = x_se_59.sigmoid()
        x_se_59 = None
        x_114 = x_113 * sigmoid_14
        x_113 = sigmoid_14 = None
        input_14 = torch.nn.functional.relu(x_114, inplace=True)
        x_114 = None
        x_115 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_6_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_6_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_6_modules_identity_parameters_bias_
        ) = None
        x_116 = torch.conv2d(
            input_14,
            l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_116 = l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_118 = torch.conv2d(
            input_14,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_14 = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_120 = x_117 + x_119
        x_117 = x_119 = None
        x_120 += x_115
        x_121 = x_120
        x_120 = x_115 = None
        x_se_60 = x_121.mean((2, 3), keepdim=True)
        x_se_61 = torch.conv2d(
            x_se_60,
            l_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_60 = l_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_62 = torch.nn.functional.relu(x_se_61, inplace=True)
        x_se_61 = None
        x_se_63 = torch.conv2d(
            x_se_62,
            l_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_62 = l_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_15 = x_se_63.sigmoid()
        x_se_63 = None
        x_122 = x_121 * sigmoid_15
        x_121 = sigmoid_15 = None
        input_15 = torch.nn.functional.relu(x_122, inplace=True)
        x_122 = None
        x_123 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_7_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_7_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_7_modules_identity_parameters_bias_
        ) = None
        x_124 = torch.conv2d(
            input_15,
            l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_7_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_126 = torch.conv2d(
            input_15,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_15 = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_128 = x_125 + x_127
        x_125 = x_127 = None
        x_128 += x_123
        x_129 = x_128
        x_128 = x_123 = None
        x_se_64 = x_129.mean((2, 3), keepdim=True)
        x_se_65 = torch.conv2d(
            x_se_64,
            l_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_64 = l_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_66 = torch.nn.functional.relu(x_se_65, inplace=True)
        x_se_65 = None
        x_se_67 = torch.conv2d(
            x_se_66,
            l_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_66 = l_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_7_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_16 = x_se_67.sigmoid()
        x_se_67 = None
        x_130 = x_129 * sigmoid_16
        x_129 = sigmoid_16 = None
        input_16 = torch.nn.functional.relu(x_130, inplace=True)
        x_130 = None
        x_131 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_8_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_8_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_8_modules_identity_parameters_bias_
        ) = None
        x_132 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_8_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_134 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_16 = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_136 = x_133 + x_135
        x_133 = x_135 = None
        x_136 += x_131
        x_137 = x_136
        x_136 = x_131 = None
        x_se_68 = x_137.mean((2, 3), keepdim=True)
        x_se_69 = torch.conv2d(
            x_se_68,
            l_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_68 = l_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_70 = torch.nn.functional.relu(x_se_69, inplace=True)
        x_se_69 = None
        x_se_71 = torch.conv2d(
            x_se_70,
            l_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_70 = l_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_8_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_17 = x_se_71.sigmoid()
        x_se_71 = None
        x_138 = x_137 * sigmoid_17
        x_137 = sigmoid_17 = None
        input_17 = torch.nn.functional.relu(x_138, inplace=True)
        x_138 = None
        x_139 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_9_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_9_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_9_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_9_modules_identity_parameters_bias_
        ) = None
        x_140 = torch.conv2d(
            input_17,
            l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_140 = l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_9_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_142 = torch.conv2d(
            input_17,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_17 = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_144 = x_141 + x_143
        x_141 = x_143 = None
        x_144 += x_139
        x_145 = x_144
        x_144 = x_139 = None
        x_se_72 = x_145.mean((2, 3), keepdim=True)
        x_se_73 = torch.conv2d(
            x_se_72,
            l_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_72 = l_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_74 = torch.nn.functional.relu(x_se_73, inplace=True)
        x_se_73 = None
        x_se_75 = torch.conv2d(
            x_se_74,
            l_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_74 = l_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_9_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_18 = x_se_75.sigmoid()
        x_se_75 = None
        x_146 = x_145 * sigmoid_18
        x_145 = sigmoid_18 = None
        input_18 = torch.nn.functional.relu(x_146, inplace=True)
        x_146 = None
        x_147 = torch.nn.functional.batch_norm(
            input_18,
            l_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_10_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_10_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_10_modules_identity_parameters_bias_
        ) = None
        x_148 = torch.conv2d(
            input_18,
            l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_10_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_150 = torch.conv2d(
            input_18,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_18 = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_151 = torch.nn.functional.batch_norm(
            x_150,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_150 = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_152 = x_149 + x_151
        x_149 = x_151 = None
        x_152 += x_147
        x_153 = x_152
        x_152 = x_147 = None
        x_se_76 = x_153.mean((2, 3), keepdim=True)
        x_se_77 = torch.conv2d(
            x_se_76,
            l_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_76 = l_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_78 = torch.nn.functional.relu(x_se_77, inplace=True)
        x_se_77 = None
        x_se_79 = torch.conv2d(
            x_se_78,
            l_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_78 = l_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_10_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_19 = x_se_79.sigmoid()
        x_se_79 = None
        x_154 = x_153 * sigmoid_19
        x_153 = sigmoid_19 = None
        input_19 = torch.nn.functional.relu(x_154, inplace=True)
        x_154 = None
        x_155 = torch.nn.functional.batch_norm(
            input_19,
            l_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_11_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_11_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_11_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_11_modules_identity_parameters_bias_
        ) = None
        x_156 = torch.conv2d(
            input_19,
            l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_157 = torch.nn.functional.batch_norm(
            x_156,
            l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_156 = l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_11_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_158 = torch.conv2d(
            input_19,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_19 = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_158 = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_160 = x_157 + x_159
        x_157 = x_159 = None
        x_160 += x_155
        x_161 = x_160
        x_160 = x_155 = None
        x_se_80 = x_161.mean((2, 3), keepdim=True)
        x_se_81 = torch.conv2d(
            x_se_80,
            l_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_80 = l_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_82 = torch.nn.functional.relu(x_se_81, inplace=True)
        x_se_81 = None
        x_se_83 = torch.conv2d(
            x_se_82,
            l_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_82 = l_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_11_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_20 = x_se_83.sigmoid()
        x_se_83 = None
        x_162 = x_161 * sigmoid_20
        x_161 = sigmoid_20 = None
        input_20 = torch.nn.functional.relu(x_162, inplace=True)
        x_162 = None
        x_163 = torch.nn.functional.batch_norm(
            input_20,
            l_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_12_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_12_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_12_modules_identity_parameters_bias_
        ) = None
        x_164 = torch.conv2d(
            input_20,
            l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_164 = l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_12_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_166 = torch.conv2d(
            input_20,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_20 = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_167 = torch.nn.functional.batch_norm(
            x_166,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_166 = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_168 = x_165 + x_167
        x_165 = x_167 = None
        x_168 += x_163
        x_169 = x_168
        x_168 = x_163 = None
        x_se_84 = x_169.mean((2, 3), keepdim=True)
        x_se_85 = torch.conv2d(
            x_se_84,
            l_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_84 = l_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_86 = torch.nn.functional.relu(x_se_85, inplace=True)
        x_se_85 = None
        x_se_87 = torch.conv2d(
            x_se_86,
            l_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_86 = l_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_12_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_21 = x_se_87.sigmoid()
        x_se_87 = None
        x_170 = x_169 * sigmoid_21
        x_169 = sigmoid_21 = None
        input_21 = torch.nn.functional.relu(x_170, inplace=True)
        x_170 = None
        x_171 = torch.nn.functional.batch_norm(
            input_21,
            l_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_13_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_13_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_13_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_13_modules_identity_parameters_bias_
        ) = None
        x_172 = torch.conv2d(
            input_21,
            l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_13_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_174 = torch.conv2d(
            input_21,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_21 = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_175 = torch.nn.functional.batch_norm(
            x_174,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_174 = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_176 = x_173 + x_175
        x_173 = x_175 = None
        x_176 += x_171
        x_177 = x_176
        x_176 = x_171 = None
        x_se_88 = x_177.mean((2, 3), keepdim=True)
        x_se_89 = torch.conv2d(
            x_se_88,
            l_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_88 = l_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_90 = torch.nn.functional.relu(x_se_89, inplace=True)
        x_se_89 = None
        x_se_91 = torch.conv2d(
            x_se_90,
            l_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_90 = l_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_13_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_22 = x_se_91.sigmoid()
        x_se_91 = None
        x_178 = x_177 * sigmoid_22
        x_177 = sigmoid_22 = None
        input_22 = torch.nn.functional.relu(x_178, inplace=True)
        x_178 = None
        x_179 = torch.conv2d(
            input_22,
            l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_179 = l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_181 = torch.conv2d(
            input_22,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_22 = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_182 = torch.nn.functional.batch_norm(
            x_181,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_181 = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_183 = x_180 + x_182
        x_180 = x_182 = None
        x_se_92 = x_183.mean((2, 3), keepdim=True)
        x_se_93 = torch.conv2d(
            x_se_92,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_92 = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_94 = torch.nn.functional.relu(x_se_93, inplace=True)
        x_se_93 = None
        x_se_95 = torch.conv2d(
            x_se_94,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_94 = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_23 = x_se_95.sigmoid()
        x_se_95 = None
        x_184 = x_183 * sigmoid_23
        x_183 = sigmoid_23 = None
        input_23 = torch.nn.functional.relu(x_184, inplace=True)
        x_184 = None
        x_185 = torch.nn.functional.batch_norm(
            input_23,
            l_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_1_modules_identity_parameters_bias_
        ) = None
        x_186 = torch.conv2d(
            input_23,
            l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_186 = l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_188 = torch.conv2d(
            input_23,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_23 = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_189 = torch.nn.functional.batch_norm(
            x_188,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_188 = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_190 = x_187 + x_189
        x_187 = x_189 = None
        x_190 += x_185
        x_191 = x_190
        x_190 = x_185 = None
        x_se_96 = x_191.mean((2, 3), keepdim=True)
        x_se_97 = torch.conv2d(
            x_se_96,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_96 = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_98 = torch.nn.functional.relu(x_se_97, inplace=True)
        x_se_97 = None
        x_se_99 = torch.conv2d(
            x_se_98,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_98 = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_24 = x_se_99.sigmoid()
        x_se_99 = None
        x_192 = x_191 * sigmoid_24
        x_191 = sigmoid_24 = None
        input_24 = torch.nn.functional.relu(x_192, inplace=True)
        x_192 = None
        x_193 = torch.nn.functional.batch_norm(
            input_24,
            l_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_2_modules_identity_parameters_bias_
        ) = None
        x_194 = torch.conv2d(
            input_24,
            l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_195 = torch.nn.functional.batch_norm(
            x_194,
            l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_194 = l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_196 = torch.conv2d(
            input_24,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_24 = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_197 = torch.nn.functional.batch_norm(
            x_196,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_196 = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_198 = x_195 + x_197
        x_195 = x_197 = None
        x_198 += x_193
        x_199 = x_198
        x_198 = x_193 = None
        x_se_100 = x_199.mean((2, 3), keepdim=True)
        x_se_101 = torch.conv2d(
            x_se_100,
            l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_100 = l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_102 = torch.nn.functional.relu(x_se_101, inplace=True)
        x_se_101 = None
        x_se_103 = torch.conv2d(
            x_se_102,
            l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_102 = l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_25 = x_se_103.sigmoid()
        x_se_103 = None
        x_200 = x_199 * sigmoid_25
        x_199 = sigmoid_25 = None
        input_25 = torch.nn.functional.relu(x_200, inplace=True)
        x_200 = None
        x_201 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_3_modules_identity_parameters_bias_
        ) = None
        x_202 = torch.conv2d(
            input_25,
            l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_203 = torch.nn.functional.batch_norm(
            x_202,
            l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_202 = l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_204 = torch.conv2d(
            input_25,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_25 = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_204 = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_206 = x_203 + x_205
        x_203 = x_205 = None
        x_206 += x_201
        x_207 = x_206
        x_206 = x_201 = None
        x_se_104 = x_207.mean((2, 3), keepdim=True)
        x_se_105 = torch.conv2d(
            x_se_104,
            l_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_104 = l_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_106 = torch.nn.functional.relu(x_se_105, inplace=True)
        x_se_105 = None
        x_se_107 = torch.conv2d(
            x_se_106,
            l_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_106 = l_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_26 = x_se_107.sigmoid()
        x_se_107 = None
        x_208 = x_207 * sigmoid_26
        x_207 = sigmoid_26 = None
        input_26 = torch.nn.functional.relu(x_208, inplace=True)
        x_208 = None
        x_209 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_4_modules_identity_parameters_bias_
        ) = None
        x_210 = torch.conv2d(
            input_26,
            l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_211 = torch.nn.functional.batch_norm(
            x_210,
            l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_210 = l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_212 = torch.conv2d(
            input_26,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_26 = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_214 = x_211 + x_213
        x_211 = x_213 = None
        x_214 += x_209
        x_215 = x_214
        x_214 = x_209 = None
        x_se_108 = x_215.mean((2, 3), keepdim=True)
        x_se_109 = torch.conv2d(
            x_se_108,
            l_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_108 = l_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_110 = torch.nn.functional.relu(x_se_109, inplace=True)
        x_se_109 = None
        x_se_111 = torch.conv2d(
            x_se_110,
            l_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_110 = l_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_27 = x_se_111.sigmoid()
        x_se_111 = None
        x_216 = x_215 * sigmoid_27
        x_215 = sigmoid_27 = None
        input_27 = torch.nn.functional.relu(x_216, inplace=True)
        x_216 = None
        x_217 = torch.nn.functional.batch_norm(
            input_27,
            l_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_5_modules_identity_parameters_bias_
        ) = None
        x_218 = torch.conv2d(
            input_27,
            l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_219 = torch.nn.functional.batch_norm(
            x_218,
            l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_218 = l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_220 = torch.conv2d(
            input_27,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_27 = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_221 = torch.nn.functional.batch_norm(
            x_220,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_220 = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_222 = x_219 + x_221
        x_219 = x_221 = None
        x_222 += x_217
        x_223 = x_222
        x_222 = x_217 = None
        x_se_112 = x_223.mean((2, 3), keepdim=True)
        x_se_113 = torch.conv2d(
            x_se_112,
            l_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_112 = l_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_114 = torch.nn.functional.relu(x_se_113, inplace=True)
        x_se_113 = None
        x_se_115 = torch.conv2d(
            x_se_114,
            l_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_114 = l_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_28 = x_se_115.sigmoid()
        x_se_115 = None
        x_224 = x_223 * sigmoid_28
        x_223 = sigmoid_28 = None
        input_28 = torch.nn.functional.relu(x_224, inplace=True)
        x_224 = None
        x_225 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_6_modules_identity_parameters_bias_
        ) = None
        x_226 = torch.conv2d(
            input_28,
            l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_227 = torch.nn.functional.batch_norm(
            x_226,
            l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_226 = l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_228 = torch.conv2d(
            input_28,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_28 = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_229 = torch.nn.functional.batch_norm(
            x_228,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_228 = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_230 = x_227 + x_229
        x_227 = x_229 = None
        x_230 += x_225
        x_231 = x_230
        x_230 = x_225 = None
        x_se_116 = x_231.mean((2, 3), keepdim=True)
        x_se_117 = torch.conv2d(
            x_se_116,
            l_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_116 = l_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_118 = torch.nn.functional.relu(x_se_117, inplace=True)
        x_se_117 = None
        x_se_119 = torch.conv2d(
            x_se_118,
            l_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_118 = l_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_29 = x_se_119.sigmoid()
        x_se_119 = None
        x_232 = x_231 * sigmoid_29
        x_231 = sigmoid_29 = None
        input_29 = torch.nn.functional.relu(x_232, inplace=True)
        x_232 = None
        x_233 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_7_modules_identity_parameters_bias_
        ) = None
        x_234 = torch.conv2d(
            input_29,
            l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_234 = l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_236 = torch.conv2d(
            input_29,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_29 = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_237 = torch.nn.functional.batch_norm(
            x_236,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_236 = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_238 = x_235 + x_237
        x_235 = x_237 = None
        x_238 += x_233
        x_239 = x_238
        x_238 = x_233 = None
        x_se_120 = x_239.mean((2, 3), keepdim=True)
        x_se_121 = torch.conv2d(
            x_se_120,
            l_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_120 = l_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_122 = torch.nn.functional.relu(x_se_121, inplace=True)
        x_se_121 = None
        x_se_123 = torch.conv2d(
            x_se_122,
            l_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_122 = l_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_30 = x_se_123.sigmoid()
        x_se_123 = None
        x_240 = x_239 * sigmoid_30
        x_239 = sigmoid_30 = None
        input_30 = torch.nn.functional.relu(x_240, inplace=True)
        x_240 = None
        x_241 = torch.nn.functional.batch_norm(
            input_30,
            l_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_8_modules_identity_parameters_bias_
        ) = None
        x_242 = torch.conv2d(
            input_30,
            l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_243 = torch.nn.functional.batch_norm(
            x_242,
            l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_242 = l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_244 = torch.conv2d(
            input_30,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_30 = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_245 = torch.nn.functional.batch_norm(
            x_244,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_244 = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_246 = x_243 + x_245
        x_243 = x_245 = None
        x_246 += x_241
        x_247 = x_246
        x_246 = x_241 = None
        x_se_124 = x_247.mean((2, 3), keepdim=True)
        x_se_125 = torch.conv2d(
            x_se_124,
            l_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_124 = l_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_126 = torch.nn.functional.relu(x_se_125, inplace=True)
        x_se_125 = None
        x_se_127 = torch.conv2d(
            x_se_126,
            l_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_126 = l_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_31 = x_se_127.sigmoid()
        x_se_127 = None
        x_248 = x_247 * sigmoid_31
        x_247 = sigmoid_31 = None
        input_31 = torch.nn.functional.relu(x_248, inplace=True)
        x_248 = None
        x_249 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_9_modules_identity_parameters_bias_
        ) = None
        x_250 = torch.conv2d(
            input_31,
            l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_251 = torch.nn.functional.batch_norm(
            x_250,
            l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_250 = l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_252 = torch.conv2d(
            input_31,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_31 = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_253 = torch.nn.functional.batch_norm(
            x_252,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_252 = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_254 = x_251 + x_253
        x_251 = x_253 = None
        x_254 += x_249
        x_255 = x_254
        x_254 = x_249 = None
        x_se_128 = x_255.mean((2, 3), keepdim=True)
        x_se_129 = torch.conv2d(
            x_se_128,
            l_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_128 = l_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_130 = torch.nn.functional.relu(x_se_129, inplace=True)
        x_se_129 = None
        x_se_131 = torch.conv2d(
            x_se_130,
            l_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_130 = l_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_32 = x_se_131.sigmoid()
        x_se_131 = None
        x_256 = x_255 * sigmoid_32
        x_255 = sigmoid_32 = None
        input_32 = torch.nn.functional.relu(x_256, inplace=True)
        x_256 = None
        x_257 = torch.nn.functional.batch_norm(
            input_32,
            l_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_10_modules_identity_parameters_bias_
        ) = None
        x_258 = torch.conv2d(
            input_32,
            l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_259 = torch.nn.functional.batch_norm(
            x_258,
            l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_258 = l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_260 = torch.conv2d(
            input_32,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_32 = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_261 = torch.nn.functional.batch_norm(
            x_260,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_260 = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_262 = x_259 + x_261
        x_259 = x_261 = None
        x_262 += x_257
        x_263 = x_262
        x_262 = x_257 = None
        x_se_132 = x_263.mean((2, 3), keepdim=True)
        x_se_133 = torch.conv2d(
            x_se_132,
            l_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_132 = l_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_134 = torch.nn.functional.relu(x_se_133, inplace=True)
        x_se_133 = None
        x_se_135 = torch.conv2d(
            x_se_134,
            l_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_134 = l_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_33 = x_se_135.sigmoid()
        x_se_135 = None
        x_264 = x_263 * sigmoid_33
        x_263 = sigmoid_33 = None
        input_33 = torch.nn.functional.relu(x_264, inplace=True)
        x_264 = None
        x_265 = torch.nn.functional.batch_norm(
            input_33,
            l_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_11_modules_identity_parameters_bias_
        ) = None
        x_266 = torch.conv2d(
            input_33,
            l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_267 = torch.nn.functional.batch_norm(
            x_266,
            l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_266 = l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_268 = torch.conv2d(
            input_33,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_33 = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_269 = torch.nn.functional.batch_norm(
            x_268,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_268 = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_270 = x_267 + x_269
        x_267 = x_269 = None
        x_270 += x_265
        x_271 = x_270
        x_270 = x_265 = None
        x_se_136 = x_271.mean((2, 3), keepdim=True)
        x_se_137 = torch.conv2d(
            x_se_136,
            l_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_136 = l_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_138 = torch.nn.functional.relu(x_se_137, inplace=True)
        x_se_137 = None
        x_se_139 = torch.conv2d(
            x_se_138,
            l_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_138 = l_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_34 = x_se_139.sigmoid()
        x_se_139 = None
        x_272 = x_271 * sigmoid_34
        x_271 = sigmoid_34 = None
        input_34 = torch.nn.functional.relu(x_272, inplace=True)
        x_272 = None
        x_273 = torch.nn.functional.batch_norm(
            input_34,
            l_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_12_modules_identity_parameters_bias_
        ) = None
        x_274 = torch.conv2d(
            input_34,
            l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_275 = torch.nn.functional.batch_norm(
            x_274,
            l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_274 = l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_276 = torch.conv2d(
            input_34,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_34 = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_277 = torch.nn.functional.batch_norm(
            x_276,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_276 = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_278 = x_275 + x_277
        x_275 = x_277 = None
        x_278 += x_273
        x_279 = x_278
        x_278 = x_273 = None
        x_se_140 = x_279.mean((2, 3), keepdim=True)
        x_se_141 = torch.conv2d(
            x_se_140,
            l_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_140 = l_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_142 = torch.nn.functional.relu(x_se_141, inplace=True)
        x_se_141 = None
        x_se_143 = torch.conv2d(
            x_se_142,
            l_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_142 = l_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_35 = x_se_143.sigmoid()
        x_se_143 = None
        x_280 = x_279 * sigmoid_35
        x_279 = sigmoid_35 = None
        input_35 = torch.nn.functional.relu(x_280, inplace=True)
        x_280 = None
        x_281 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_13_modules_identity_parameters_bias_
        ) = None
        x_282 = torch.conv2d(
            input_35,
            l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_283 = torch.nn.functional.batch_norm(
            x_282,
            l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_282 = l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_284 = torch.conv2d(
            input_35,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_35 = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_285 = torch.nn.functional.batch_norm(
            x_284,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_284 = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_286 = x_283 + x_285
        x_283 = x_285 = None
        x_286 += x_281
        x_287 = x_286
        x_286 = x_281 = None
        x_se_144 = x_287.mean((2, 3), keepdim=True)
        x_se_145 = torch.conv2d(
            x_se_144,
            l_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_144 = l_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_146 = torch.nn.functional.relu(x_se_145, inplace=True)
        x_se_145 = None
        x_se_147 = torch.conv2d(
            x_se_146,
            l_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_146 = l_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_36 = x_se_147.sigmoid()
        x_se_147 = None
        x_288 = x_287 * sigmoid_36
        x_287 = sigmoid_36 = None
        input_36 = torch.nn.functional.relu(x_288, inplace=True)
        x_288 = None
        x_289 = torch.nn.functional.batch_norm(
            input_36,
            l_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_14_modules_identity_parameters_bias_
        ) = None
        x_290 = torch.conv2d(
            input_36,
            l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_291 = torch.nn.functional.batch_norm(
            x_290,
            l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_290 = l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_292 = torch.conv2d(
            input_36,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_36 = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_293 = torch.nn.functional.batch_norm(
            x_292,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_292 = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_294 = x_291 + x_293
        x_291 = x_293 = None
        x_294 += x_289
        x_295 = x_294
        x_294 = x_289 = None
        x_se_148 = x_295.mean((2, 3), keepdim=True)
        x_se_149 = torch.conv2d(
            x_se_148,
            l_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_148 = l_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_150 = torch.nn.functional.relu(x_se_149, inplace=True)
        x_se_149 = None
        x_se_151 = torch.conv2d(
            x_se_150,
            l_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_150 = l_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_37 = x_se_151.sigmoid()
        x_se_151 = None
        x_296 = x_295 * sigmoid_37
        x_295 = sigmoid_37 = None
        input_37 = torch.nn.functional.relu(x_296, inplace=True)
        x_296 = None
        x_297 = torch.nn.functional.batch_norm(
            input_37,
            l_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_15_modules_identity_parameters_bias_
        ) = None
        x_298 = torch.conv2d(
            input_37,
            l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_299 = torch.nn.functional.batch_norm(
            x_298,
            l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_298 = l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_300 = torch.conv2d(
            input_37,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_37 = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_301 = torch.nn.functional.batch_norm(
            x_300,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_300 = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_302 = x_299 + x_301
        x_299 = x_301 = None
        x_302 += x_297
        x_303 = x_302
        x_302 = x_297 = None
        x_se_152 = x_303.mean((2, 3), keepdim=True)
        x_se_153 = torch.conv2d(
            x_se_152,
            l_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_152 = l_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_154 = torch.nn.functional.relu(x_se_153, inplace=True)
        x_se_153 = None
        x_se_155 = torch.conv2d(
            x_se_154,
            l_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_154 = l_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_38 = x_se_155.sigmoid()
        x_se_155 = None
        x_304 = x_303 * sigmoid_38
        x_303 = sigmoid_38 = None
        input_38 = torch.nn.functional.relu(x_304, inplace=True)
        x_304 = None
        x_305 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_16_modules_identity_parameters_bias_
        ) = None
        x_306 = torch.conv2d(
            input_38,
            l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_307 = torch.nn.functional.batch_norm(
            x_306,
            l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_306 = l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_308 = torch.conv2d(
            input_38,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_38 = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_309 = torch.nn.functional.batch_norm(
            x_308,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_308 = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_310 = x_307 + x_309
        x_307 = x_309 = None
        x_310 += x_305
        x_311 = x_310
        x_310 = x_305 = None
        x_se_156 = x_311.mean((2, 3), keepdim=True)
        x_se_157 = torch.conv2d(
            x_se_156,
            l_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_156 = l_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_158 = torch.nn.functional.relu(x_se_157, inplace=True)
        x_se_157 = None
        x_se_159 = torch.conv2d(
            x_se_158,
            l_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_158 = l_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_39 = x_se_159.sigmoid()
        x_se_159 = None
        x_312 = x_311 * sigmoid_39
        x_311 = sigmoid_39 = None
        input_39 = torch.nn.functional.relu(x_312, inplace=True)
        x_312 = None
        x_313 = torch.nn.functional.batch_norm(
            input_39,
            l_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_17_modules_identity_parameters_bias_
        ) = None
        x_314 = torch.conv2d(
            input_39,
            l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_315 = torch.nn.functional.batch_norm(
            x_314,
            l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_314 = l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_316 = torch.conv2d(
            input_39,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_39 = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_317 = torch.nn.functional.batch_norm(
            x_316,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_316 = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_318 = x_315 + x_317
        x_315 = x_317 = None
        x_318 += x_313
        x_319 = x_318
        x_318 = x_313 = None
        x_se_160 = x_319.mean((2, 3), keepdim=True)
        x_se_161 = torch.conv2d(
            x_se_160,
            l_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_160 = l_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_162 = torch.nn.functional.relu(x_se_161, inplace=True)
        x_se_161 = None
        x_se_163 = torch.conv2d(
            x_se_162,
            l_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_162 = l_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_40 = x_se_163.sigmoid()
        x_se_163 = None
        x_320 = x_319 * sigmoid_40
        x_319 = sigmoid_40 = None
        input_40 = torch.nn.functional.relu(x_320, inplace=True)
        x_320 = None
        x_321 = torch.nn.functional.batch_norm(
            input_40,
            l_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_18_modules_identity_parameters_bias_
        ) = None
        x_322 = torch.conv2d(
            input_40,
            l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_323 = torch.nn.functional.batch_norm(
            x_322,
            l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_322 = l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_324 = torch.conv2d(
            input_40,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_40 = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_325 = torch.nn.functional.batch_norm(
            x_324,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_324 = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_326 = x_323 + x_325
        x_323 = x_325 = None
        x_326 += x_321
        x_327 = x_326
        x_326 = x_321 = None
        x_se_164 = x_327.mean((2, 3), keepdim=True)
        x_se_165 = torch.conv2d(
            x_se_164,
            l_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_164 = l_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_166 = torch.nn.functional.relu(x_se_165, inplace=True)
        x_se_165 = None
        x_se_167 = torch.conv2d(
            x_se_166,
            l_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_166 = l_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_41 = x_se_167.sigmoid()
        x_se_167 = None
        x_328 = x_327 * sigmoid_41
        x_327 = sigmoid_41 = None
        input_41 = torch.nn.functional.relu(x_328, inplace=True)
        x_328 = None
        x_329 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_19_modules_identity_parameters_bias_
        ) = None
        x_330 = torch.conv2d(
            input_41,
            l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_331 = torch.nn.functional.batch_norm(
            x_330,
            l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_330 = l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_332 = torch.conv2d(
            input_41,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_41 = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_333 = torch.nn.functional.batch_norm(
            x_332,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_332 = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_334 = x_331 + x_333
        x_331 = x_333 = None
        x_334 += x_329
        x_335 = x_334
        x_334 = x_329 = None
        x_se_168 = x_335.mean((2, 3), keepdim=True)
        x_se_169 = torch.conv2d(
            x_se_168,
            l_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_168 = l_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_170 = torch.nn.functional.relu(x_se_169, inplace=True)
        x_se_169 = None
        x_se_171 = torch.conv2d(
            x_se_170,
            l_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_170 = l_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_42 = x_se_171.sigmoid()
        x_se_171 = None
        x_336 = x_335 * sigmoid_42
        x_335 = sigmoid_42 = None
        input_42 = torch.nn.functional.relu(x_336, inplace=True)
        x_336 = None
        x_337 = torch.nn.functional.batch_norm(
            input_42,
            l_self_modules_stages_modules_2_modules_20_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_20_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_20_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_20_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_20_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_20_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_20_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_20_modules_identity_parameters_bias_
        ) = None
        x_338 = torch.conv2d(
            input_42,
            l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_339 = torch.nn.functional.batch_norm(
            x_338,
            l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_338 = l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_20_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_340 = torch.conv2d(
            input_42,
            l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_42 = l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_341 = torch.nn.functional.batch_norm(
            x_340,
            l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_340 = l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_20_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_342 = x_339 + x_341
        x_339 = x_341 = None
        x_342 += x_337
        x_343 = x_342
        x_342 = x_337 = None
        x_se_172 = x_343.mean((2, 3), keepdim=True)
        x_se_173 = torch.conv2d(
            x_se_172,
            l_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_172 = l_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_174 = torch.nn.functional.relu(x_se_173, inplace=True)
        x_se_173 = None
        x_se_175 = torch.conv2d(
            x_se_174,
            l_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_174 = l_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_20_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_43 = x_se_175.sigmoid()
        x_se_175 = None
        x_344 = x_343 * sigmoid_43
        x_343 = sigmoid_43 = None
        input_43 = torch.nn.functional.relu(x_344, inplace=True)
        x_344 = None
        x_345 = torch.nn.functional.batch_norm(
            input_43,
            l_self_modules_stages_modules_2_modules_21_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_21_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_21_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_21_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_21_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_21_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_21_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_21_modules_identity_parameters_bias_
        ) = None
        x_346 = torch.conv2d(
            input_43,
            l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_347 = torch.nn.functional.batch_norm(
            x_346,
            l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_346 = l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_21_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_348 = torch.conv2d(
            input_43,
            l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_43 = l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_349 = torch.nn.functional.batch_norm(
            x_348,
            l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_348 = l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_21_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_350 = x_347 + x_349
        x_347 = x_349 = None
        x_350 += x_345
        x_351 = x_350
        x_350 = x_345 = None
        x_se_176 = x_351.mean((2, 3), keepdim=True)
        x_se_177 = torch.conv2d(
            x_se_176,
            l_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_176 = l_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_178 = torch.nn.functional.relu(x_se_177, inplace=True)
        x_se_177 = None
        x_se_179 = torch.conv2d(
            x_se_178,
            l_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_178 = l_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_21_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_44 = x_se_179.sigmoid()
        x_se_179 = None
        x_352 = x_351 * sigmoid_44
        x_351 = sigmoid_44 = None
        input_44 = torch.nn.functional.relu(x_352, inplace=True)
        x_352 = None
        x_353 = torch.nn.functional.batch_norm(
            input_44,
            l_self_modules_stages_modules_2_modules_22_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_22_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_22_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_22_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_22_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_22_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_22_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_22_modules_identity_parameters_bias_
        ) = None
        x_354 = torch.conv2d(
            input_44,
            l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_355 = torch.nn.functional.batch_norm(
            x_354,
            l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_354 = l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_22_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_356 = torch.conv2d(
            input_44,
            l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_44 = l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_357 = torch.nn.functional.batch_norm(
            x_356,
            l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_356 = l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_22_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_358 = x_355 + x_357
        x_355 = x_357 = None
        x_358 += x_353
        x_359 = x_358
        x_358 = x_353 = None
        x_se_180 = x_359.mean((2, 3), keepdim=True)
        x_se_181 = torch.conv2d(
            x_se_180,
            l_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_180 = l_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_182 = torch.nn.functional.relu(x_se_181, inplace=True)
        x_se_181 = None
        x_se_183 = torch.conv2d(
            x_se_182,
            l_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_182 = l_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_22_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_45 = x_se_183.sigmoid()
        x_se_183 = None
        x_360 = x_359 * sigmoid_45
        x_359 = sigmoid_45 = None
        input_45 = torch.nn.functional.relu(x_360, inplace=True)
        x_360 = None
        x_361 = torch.nn.functional.batch_norm(
            input_45,
            l_self_modules_stages_modules_2_modules_23_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_23_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_23_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_23_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_23_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_23_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_23_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_23_modules_identity_parameters_bias_
        ) = None
        x_362 = torch.conv2d(
            input_45,
            l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_363 = torch.nn.functional.batch_norm(
            x_362,
            l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_362 = l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_23_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_364 = torch.conv2d(
            input_45,
            l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_45 = l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_365 = torch.nn.functional.batch_norm(
            x_364,
            l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_364 = l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_23_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_366 = x_363 + x_365
        x_363 = x_365 = None
        x_366 += x_361
        x_367 = x_366
        x_366 = x_361 = None
        x_se_184 = x_367.mean((2, 3), keepdim=True)
        x_se_185 = torch.conv2d(
            x_se_184,
            l_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_184 = l_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_186 = torch.nn.functional.relu(x_se_185, inplace=True)
        x_se_185 = None
        x_se_187 = torch.conv2d(
            x_se_186,
            l_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_186 = l_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_23_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_46 = x_se_187.sigmoid()
        x_se_187 = None
        x_368 = x_367 * sigmoid_46
        x_367 = sigmoid_46 = None
        input_46 = torch.nn.functional.relu(x_368, inplace=True)
        x_368 = None
        x_369 = torch.conv2d(
            input_46,
            l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_370 = torch.nn.functional.batch_norm(
            x_369,
            l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_369 = l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_371 = torch.conv2d(
            input_46,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_46 = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_372 = torch.nn.functional.batch_norm(
            x_371,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_371 = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_373 = x_370 + x_372
        x_370 = x_372 = None
        x_se_188 = x_373.mean((2, 3), keepdim=True)
        x_se_189 = torch.conv2d(
            x_se_188,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_188 = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc1_parameters_bias_ = (None)
        x_se_190 = torch.nn.functional.relu(x_se_189, inplace=True)
        x_se_189 = None
        x_se_191 = torch.conv2d(
            x_se_190,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_190 = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_fc2_parameters_bias_ = (None)
        sigmoid_47 = x_se_191.sigmoid()
        x_se_191 = None
        x_374 = x_373 * sigmoid_47
        x_373 = sigmoid_47 = None
        input_47 = torch.nn.functional.relu(x_374, inplace=True)
        x_374 = None
        x_375 = torch.nn.functional.adaptive_avg_pool2d(input_47, 1)
        input_47 = None
        x_376 = x_375.flatten(1, -1)
        x_375 = None
        x_377 = torch.nn.functional.dropout(x_376, 0.0, False, False)
        x_376 = None
        x_378 = torch._C._nn.linear(
            x_377,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_377 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_378,)
